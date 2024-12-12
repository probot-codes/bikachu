from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
import numpy as np
import instaloader
import pickle
from flask_cors import CORS  # Import CORS
import pandas as pd  # Import pandas for DataFrame
import cv2  # Import OpenCV for image comparison
import requests  # For fetching images from URLs
import tempfile  # For temporary file handling
import requests
import json
from datetime import datetime
from bs4 import BeautifulSoup
import gugl
import foto

load_dotenv()
app = Flask(__name__)
USERNAME = os.getenv('USERNAME')
PASSWD = os.getenv('PASSWD')

CORS(app)  # Enable CORS for all routes

# Load the model and scaler
with open('./model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Create the Instaloader instance once
loader = instaloader.Instaloader()
COOKIES_PATH = "cookies.txt"

try:
    if os.path.exists(COOKIES_PATH):
        loader.load_session_from_file(USERNAME, COOKIES_PATH)  # Load session from file
        print("Loaded session from cookies.")
    else:
        loader.login(USERNAME, PASSWD)  # Replace with your actual credentials
        loader.save_session_to_file(COOKIES_PATH)  # Save session to file
        print("Logged in and saved session to cookies.")
except Exception as e:
    print(f"Error during login or cookie handling: {e}")

# Global variable to store the prediction result
is_default_profile_pic = False

def has_custom_profile_pic(profile):
    """Checks if the profile has a custom profile picture."""
    global is_default_profile_pic
    try:
        # Load default profile pictures
        default_pics = [cv2.imread("igdefault.jpg"), cv2.imread("ig2.jpg")]
        if any(pic is None for pic in default_pics):
            raise FileNotFoundError("Default profile picture files are missing or corrupted.")

        # Fetch the profile picture
        response = requests.get(profile.profile_pic_url, stream=True)
        if response.status_code != 200:
            print(f"Failed to download profile picture. HTTP status: {response.status_code}")
            return True  # Assume custom picture on failure

        # Save the profile picture temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            for chunk in response.iter_content(1024):  # Stream content to avoid memory overflow
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        # Load the saved profile picture
        profile_pic = cv2.imread(tmp_file_path)
        os.remove(tmp_file_path)  # Clean up the temporary file

        if profile_pic is None:
            print("Error: Failed to load profile picture with OpenCV.")
            return True  # Assume custom picture on error

        # Compare the profile picture with default pictures
        is_default_profile_pic = any(
            pic.shape == profile_pic.shape and not np.bitwise_xor(pic, profile_pic).any()
            for pic in default_pics if pic is not None
        )
        return not is_default_profile_pic  # Return True if custom, False if default

    except Exception as e:
        print(f"Error comparing profile pictures: {e}")
        return True  # Assume custom picture on failure


def extract_features_instaloader(username):
    """Extracts features and profile information from an Instagram profile."""
    try:
        profile = instaloader.Profile.from_username(loader.context, username)

        # Extract feature vector
        profile_pic = 1 if has_custom_profile_pic(profile) else 0
        nums_length_username = sum(c.isdigit() for c in profile.username) / len(profile.username)
        fullname_words = len(profile.full_name.split())
        nums_length_fullname = sum(c.isdigit() for c in profile.full_name) / max(len(profile.full_name), 1)
        name_equals_username = 1 if profile.full_name.replace(" ", "").lower() == profile.username.lower() else 0
        description_length = len(profile.biography)
        external_url = 1 if profile.external_url else 0
        private = 1 if profile.is_private else 0
        num_posts = profile.mediacount
        num_followers = profile.followers
        num_follows = profile.followees

        activity_ratio = np.round(num_posts / num_followers, 2) if num_followers else 0
        followers_gt_follows = 1 if num_followers > num_follows else 0

        features = [
            profile_pic, nums_length_username, fullname_words,
            nums_length_fullname, name_equals_username, description_length,
            external_url, private, num_posts, num_followers,
            num_follows, activity_ratio, followers_gt_follows
        ]

        profile_info = {
            "username": profile.username,
            "full_name": profile.full_name,
            "biography": profile.biography,
            "profile_pic_url": profile.profile_pic_url,
            "is_private": profile.is_private,
            "num_posts": profile.mediacount,
            "num_followers": profile.followers,
            "num_follows": profile.followees,
            "external_url": profile.external_url,
        }

        return features, profile_info

    except instaloader.exceptions.ProfileNotExistsException:
        print("Error: Profile does not exist.")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get username from the request
        data = request.get_json()
        username = data.get('username')

        if not username:
            return jsonify({'error': 'Username is required'}), 400

        # Extract features using instaloader
        features, profile_info = extract_features_instaloader(username)
        if profile_info is None:
            return jsonify({'error': 'Failed to fetch profile information'}), 500

        # Prepare features for the model
        feature_names = ['profile pic', 'nums/length username', 'fullname words',
                         'nums/length fullname', 'name==username', 'description length',
                         'external URL', 'private', '#posts', '#followers', '#follows',
                         'activity ratio', '#followers > #follows?']
        input_data = pd.DataFrame([features], columns=feature_names)
        scaled_input_data = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(scaled_input_data)
        fake_probability = float(model.predict_proba(scaled_input_data)[:, 1])

        response_data = {
            'fake_probability': fake_probability,
            'is_fake': bool(prediction[0]),
            'profile_info': profile_info
        }

        print("Sent data (without social links):", response_data)
        return jsonify(response_data)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/social_links', methods=['POST'])
def get_social_links():
    try:
        data = request.get_json()
        username = data.get('username')

        if not username:
            return jsonify({'error': 'Username is required'}), 400

        social_links = check_social_media_presence(username)
        return jsonify({'social_links': social_links})

    except Exception as e:
        print(f"Error getting social links: {e}")
        return jsonify({'error': str(e)}), 500


def check_social_media_presence(username):
    """
    Checks if an account with the given username exists on various platforms.
    """
    platforms = {
        "Facebook": "https://www.facebook.com/{}",
        "Twitter": "https://x.com/{}",
        "Instagram": "https://www.instagram.com/{}",
        "Pinterest": "https://www.pinterest.com/{}",
        "LinkedIn": "https://www.linkedin.com/in/{}",
        "YouTube": "https://www.youtube.com/@{}", 
        "Tumblr": "https://{}.tumblr.com/",
        "Reddit": "https://www.reddit.com/user/{}",
        "Medium": "https://medium.com/@{}",
        "GitHub": "https://github.com/{}",
        "GitLab": "https://gitlab.com/{}",
        "Bitbucket": "https://bitbucket.org/{}/",
        "Quora": "https://www.quora.com/profile/{}",
        # "Discord": "https://discord.com/users/{}",  # Requires Discord User ID 
    }
    found_links = []

    for platform, url_template in platforms.items():
        url = url_template.format(username)
        try:
            response = requests.get(url)
            # Platform-specific checks for profile existence
            if response.status_code == 200:
                if platform == "Facebook" and "This content isn't available at the moment" not in response.text:
                    found_links.append({"platform": platform, "url": url})
                elif platform == "Twitter" and "This account doesn’t exist" not in response.text: 
                    found_links.append({"platform": platform, "url": url})
                elif platform == "Instagram" and "Follow" in response.text:
                    found_links.append({"platform": platform, "url": url})
                elif platform == "Pinterest" and "Here’s how it works." not in response.text:
                    found_links.append({"platform": platform, "url": url})
                elif platform == "YouTube" and "This page isn't available." not in response.text:
                    found_links.append({"platform": platform, "url": url})
                elif platform == "Tumblr" and "There's nothing here." not in response.text:
                    found_links.append({"platform": platform, "url": url})
                elif platform == "Reddit":
                    if "This account has been suspended" not in response.text  and "Sorry, nobody on Reddit" not in response.text:
                        found_links.append({"platform": platform, "url": url})
                    else:
                        found_links.append({"platform": platform, "url": url})
                elif platform == "Medium" and "Out of nothing, something." not in response.text: 
                    found_links.append({"platform": platform, "url": url}) 
                elif platform == "GitHub" and "Find code, projects, and people on GitHub:" not in response.text:
                    found_links.append({"platform": platform, "url": url})
                elif platform == "Bitbucket" and "Resource not found" not in response.text:
                    found_links.append({"platform": platform, "url": url})
                elif platform == "Quora" and "Page Not Found" not in response.text:  
                    found_links.append({"platform": platform, "url": url})
                elif platform == "Twitch" and "Sorry. Unless you've got a time machine, that content is unavailable." not in response.text:
                    found_links.append({"platform": platform, "url": url}) 
                # ... Add checks for other platforms ...


        except requests.exceptions.RequestException as e:
            print(f"Error checking {platform}: {e}")
            continue

    return found_links

@app.route('/reverse_search', methods=['POST'])
def reverse_search():
    """
    Endpoint to perform reverse image search.
    """
    try:
        data = request.get_json()
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({'error': 'Image URL is required'}), 400

        results = foto.reverse_image_search(image_url)  # Call the function from foto.py
        return jsonify(results)  # Return the results as JSON

    except Exception as e:
        print(f"Error during reverse image search: {e}")
        return jsonify({'error': str(e)}), 500
    
def fetch_url_content(url, save_path):
    """
    Fetch the HTML content from the given URL and save it to a file.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        with open(save_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"HTML content saved to {save_path}")

        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL content: {e}")
        return None

def extract_features_twitter(html_content, thtml_content, username):
    """
    Extract specified features from the HTML content.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    def safe_find(selector):
        """
        Safely find an element by CSS selector and return its text or None if not found.
        """
        element = soup.select_one(selector)
        return element.text.strip() if element else None

    def extract_stat(label):
        """
        Extract numerical stats like followers, friends, likes, and tweets.
        """
        stat_label = soup.find("span", string=label)
        if stat_label:
            stat_value = stat_label.find_next("span", class_="profile-stat-num")
            return int(stat_value.text.replace(",", "")) if stat_value else None
        return None

    # Find the most recent tweet
    recent_tweet = soup.find("div", class_="timeline-item")

    # Extract features
    profile_data = {
        "id": None,
        "id_str": None,
        "screen_name": safe_find(".profile-card-username"),
        "location": safe_find(".profile-location span:last-child"),  # Updated selector
        "description": safe_find(".profile-bio p"),
        "url": "https://x.com/" + safe_find(".profile-card-username") if safe_find(".profile-card-username") else None,  # Construct URL dynamically
        "followers_count": extract_stat("Followers"),
        "friends_count": extract_stat("Following"),
        "listed_count": None,
        "created_at": datetime.strptime(
            soup.find("div", class_="profile-joindate").find("span", title=True)["title"], 
            "%I:%M %p - %d %b %Y"
        ).strftime("%a %b %d %H:%M") if soup.find("div", class_="profile-joindate") else None,  # Format datetime
        "favorites_count": extract_stat("Likes"),
        "verified": bool(soup.find("span", class_="verified-icon")),
        "statuses_count": extract_stat("Tweets"),
        "lang": "en", 
        "status": recent_tweet.find('div', class_='tweet-content media-body').get_text(strip=True) if recent_tweet else None,  # Extract recent tweet content
        "avatar_image": soup.select_one(".profile-card-avatar img")['src'] if soup.select_one(".profile-card-avatar img") else None,
        "default_profile": None,
        "default_profile_image": (
            soup.select_one(".profile-card-avatar img")['src'] == "/pic/abs.twimg.com%2Fsticky%2Fdefault_profile_images%2Fdefault_profile_400x400.png"
            if soup.select_one(".profile-card-avatar img")
            else False
        ),
        "banner_image": soup.select_one(".profile-banner img")['src'] if soup.select_one(".profile-banner img") else None,
        "first_tweet_date": str(recent_tweet.find('span', class_='tweet-date')) if recent_tweet else None,
        "has_extended_profile": None,
        "name": safe_find(".profile-card-fullname"),
        # Corrected values
        "tweet_content": recent_tweet.find('div', class_='tweet-content media-body').get_text(strip=True) if recent_tweet else None,
        "tweet_date_element": str(recent_tweet.find('span', class_='tweet-date')) if recent_tweet else None,  # Already converted to string
        "tweet_timestamp": recent_tweet.find('span', class_='tweet-date a')['title'] if recent_tweet and recent_tweet.find('span', class_='tweet-date a') else "Unknown time",
    }

    return profile_data

@app.route('/predict_twitter', methods=['POST'])
def predict_twitter():
    try:
        # Get username from the request
        data = request.get_json()
        username = data.get('username')

        if not username:
            return jsonify({'error': 'Username is required'}), 400

        # Construct URLs
        url = "https://nitter.privacydev.net/" + username
        turl = "https://x.com/" + username

        # Fetch HTML content
        html_content = fetch_url_content(url, "./output.html")
        thtml_content = fetch_url_content(turl, "./toutput.html")

        if html_content:
            # Extract features
            profile_info = extract_features_twitter(html_content, thtml_content, username)

            # --- Heuristic to estimate fake probability ---
            fake_probability = calculate_fake_probability(profile_info)

            response_data = {
                'fake_probability': fake_probability,  # Return as 'fake_probability'
                'is_fake': fake_probability > 0.5,  # Determine 'is_fake' based on threshold
                'profile_info': profile_info
            }
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Failed to fetch profile information'}), 500

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

def calculate_fake_probability(profile_info):
    """
    Calculates an approximate fake probability based on Twitter profile features.
    """
    try:
        # Initialize score
        score = 0

        # --- More comprehensive checks ---

        # Account creation date
        if profile_info['created_at'] is not None:
            account_age_days = (datetime.now() - datetime.strptime(profile_info['created_at'], "%a %b %d %H:%M")).days
            if account_age_days < 30:  
                score += 3
            elif account_age_days < 90:
                score += 1

        # Number of followers
        if profile_info['followers_count'] < 10:  
            score += 3
        elif profile_info['followers_count'] < 100:
            score += 1

        # Number of friends (following)
        if profile_info['friends_count'] > 1000:  
            score += 2
        elif profile_info['friends_count'] > 500:
            score += 1

        # Number of tweets
        if profile_info['statuses_count'] < 10:  
            score += 3
        elif profile_info['statuses_count'] < 50:
            score += 1

        # Profile image
        if profile_info['default_profile_image']:  
            score += 2

        # Profile description
        if profile_info['description'] is None or len(profile_info['description']) < 10:  
            score += 2

        # Ratio of followers to friends
        if profile_info['followers_count'] and profile_info['friends_count']:
            if profile_info['followers_count'] / profile_info['friends_count'] < 0.1:
                score += 2

        # Check for suspicious keywords in username or description (example)
        suspicious_keywords = ["bot", "promo", "cheap", "follow", "like"]
        if any(keyword in profile_info['screen_name'].lower() for keyword in suspicious_keywords):
            score += 2
        if any(keyword in profile_info['description'].lower() for keyword in suspicious_keywords):
            score += 1

        # ... add more conditions based on other features ...

        # Normalize score to a probability (0 to 1)
        probability = min(score / 20, 1)  # Adjust the divisor as needed 

        return probability

    except Exception as e:
        print(f"Error calculating fake probability: {e}")
        return 0.5  # Return a neutral probability on error


@app.route('/google_search', methods=['POST'])
def google_search_endpoint():
    try:
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        results = gugl.gugl_search(query)

        # Print the results to the console
        print("Google Search Results:")
        for result in results:
            print(result)

        return jsonify({'results': results})

    except Exception as e:
        print(f"Error in google_search_endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
