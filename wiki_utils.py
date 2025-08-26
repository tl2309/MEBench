# -*- encoding: utf-8 -*-

import requests

class wikiutils(object):

    def __init__(self, **kwargs):
        self.url = "https://www.wikidata.org/w/api.php"

    def get_wikidata_id(self, search_term):
        """Fetch the Wikidata identifier for a given search term."""
        # url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json",
            "search": search_term
        }

        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()
            data = response.json()

            if data['search']:
                return data['search'][0]['id']
            else:
                return None
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def name_belongs_to_entity(self, wikidata_id, topicid):
        """Checks if the entity with the given Wikidata ID is an ACM Fellow."""
        query = """
        SELECT ?award WHERE {
          wd:%s p:P166 ?statement.
          ?statement ps:P166 ?award.
          FILTER(?award = wd:%s)
        }
        """ % (wikidata_id, topicid)

        url = "https://query.wikidata.org/sparql"
        headers = {"Accept": "application/sparql-results+json"}
        params = {"query": query, "format": "json"}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            return len(data['results']['bindings']) > 0
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return False

    # Example usage
    # name = "C. William Gear"  # Replace with the name you're interested in
    # wikidata_id = get_wikidata_id(name)
    # topic = "ACM fellow"
    # topicid = get_wikidata_id(topic)
    # print(topicid)
    # if wikidata_id:
    #     if name_belongs_to_entity(wikidata_id, topicid):
    #         print(f"{name} (Wikidata ID: {wikidata_id}) is  {topic}.")
    #     else:
    #         print(f"{name} (Wikidata ID: {wikidata_id}) is not {topic}.")
    # else:
    #     print(f"No Wikidata ID found for '{name}'.")

    def search_wikidata_property(self, keyword):
        """
        Search for a Wikidata property ID based on a keyword.
        Returns a list of matching properties with their IDs and descriptions.
        """
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "language": "en",
            "type": "property",
            "format": "json",
            "search": keyword
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raises an HTTPError if the response status code is 4XX/5XX
            data = response.json()

            properties = []
            prop_id = ''
            for result in data.get("search", []):
                prop_id = result.get("id")
                label = result.get("label")
                description = result.get("description", "No description available.")
                properties.append((prop_id, label, description))

            return properties
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return []

    def get_entities_wikipedia_titles(self, property_id, value_id):
        """Fetches Wikipedia titles for entities based on a specified property and its value."""
        query = f"""
          SELECT ?entity ?entityLabel ?article WHERE {{
          ?entity p:{property_id} ?statement.
          ?statement ps:{property_id} wd:{value_id}.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
          OPTIONAL {{
            ?article schema:about ?entity;
                     schema:inLanguage "en";
                     schema:isPartOf <https://en.wikipedia.org/>.
          }}
        }}
        """
        url = "https://query.wikidata.org/sparql"
        headers = {"Accept": "application/sparql-results+json"}
        params = {"query": query}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            titles = []
            for item in data['results']['bindings']:
                if 'article' in item:
                    article_url = item['article']['value']
                    title = article_url.split('/')[-1]  # Extract Wikipedia title from URL
                    titles.append(title)

            return titles
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return []

    def get_wikipedia_first_paragraph(self, title):
        """Fetches the first paragraph of a Wikipedia page for a given title."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            page = next(iter(data['query']['pages'].values()))
            if 'extract' in page:
                content = page['extract']
                first_paragraph = content.split('\n')[0]  # Assuming the first paragraph is before the first newline
                return first_paragraph
            else:
                return "Content not available."
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return "Error retrieving content."
