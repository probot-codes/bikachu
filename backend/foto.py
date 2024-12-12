from google_img_source_search import ReverseImageSearcher
import json

def reverse_image_search(image_url, max_results=10):
    # Create an instance of ReverseImageSearcher
    rev_img_searcher = ReverseImageSearcher()
    
    # Perform the reverse image search
    results = rev_img_searcher.search(image_url)
    
    # Check if results were returned
    if results:
        output = {'results': []}  # Initialize JSON output
        filtered_results = []

        # Filter results for Twitter and Instagram
        for search_item in results:
            if 'twitter.com' in search_item.page_url or 'instagram.com' in search_item.page_url:
                filtered_results.append(search_item)

        # Prepare JSON output
        for index, search_item in enumerate(filtered_results):
            if index < max_results:
                output['results'].append({
                    'site': search_item.page_url,
                    'image': search_item.image_url
                })
        
        if not filtered_results:
            output['message'] = 'No results found for Twitter or Instagram.'
        
        # Print JSON output
        print(json.dumps(output, indent=4))
    else:
        # Print message for no results
        print(json.dumps({'message': 'No results found.'}, indent=4))

if __name__ == "__main__":
    # Input: Image URL
    image_url = input("Enter the image URL: ")
    reverse_image_search(image_url)
