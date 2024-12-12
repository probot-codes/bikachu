from googlesearch import search

def google_search(query, domain=None, num_results=10):
    results = []
    if domain:
        if domain == "x.com":
            search_query = f"{query} twitter"
        else:
            search_query = f"{query} site:{domain}"
    else:
        search_query = query  # Use the query directly if no domain is specified

    # print(f"Debug: search_query for {domain} is: {search_query}") 

    for result in search(search_query, num_results=num_results):
        results.append(result)

    return results

def gugl_search(query):
    domains = ["x.com", "github.com", "instagram.com", "linkedin.com"]
    all_results = []
    for domain in domains:
        results = google_search(query, domain, num_results=5)  # Limit to 5 results per domain
        all_results.extend(results)
    return all_results

def main():
    query_string = input("Enter your search query: ").strip()
    domains = ["x.com", "github.com", "instagram.com", "linkedin.com"]
    
    num_results = 10

    for domain in domains:
        print(f"\nResults from {domain}:")
        results = google_search(query_string, domain, num_results)
        for link in results:
            print(link)

if __name__ == "__main__":
    main()