
import os
import json
import logging
from typing import List, Dict, Optional
from firecrawl import FirecrawlApp
from urllib.parse import urlparse
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_DIR = "scraped_content"

mcp = FastMCP("llm_inference")

@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] = ['markdown', 'html'],
    api_key: Optional[str] = None
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.
    
    Args:
        websites: Dictionary of provider_name -> URL mappings
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)
        
    Returns:
        List of provider names for successfully scraped websites
    """
    
    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")
    
    app = FirecrawlApp(api_key=api_key)
    
    path = os.path.join(SCRAPE_DIR)
    os.makedirs(path, exist_ok=True)
    
    # save the scraped content to files and then create scraped_metadata.json as a summary file
    # check if the provider has already been scraped and decide if you want to overwrite
    # {
    #     "cloudrift_ai": {
    #         "provider_name": "cloudrift_ai",
    #         "url": "https://www.cloudrift.ai/inference",
    #         "domain": "www.cloudrift.ai",
    #         "scraped_at": "2025-10-23T00:44:59.902569",
    #         "formats": [
    #             "markdown",
    #             "html"
    #         ],
    #         "success": "true",
    #         "content_files": {
    #             "markdown": "cloudrift_ai_markdown.txt",
    #             "html": "cloudrift_ai_html.txt"
    #         },
    #         "title": "AI Inference",
    #         "description": "Scraped content goes here"
    #     }
    # }
    metadata_file = os.path.join(path, "scraped_metadata.json")

    # continue your solution here ...
    scraped_metadata: Dict = {}
    try:
        if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
            with open(metadata_file, "r", encoding="utf-8") as mf:
                scraped_metadata = json.load(mf) or {}
        else:
            scraped_metadata = {}
    except Exception as e:
        logger.warning(f"Unable to load metadata: {e}")
        scraped_metadata = {}

    successful_scrapes = [key for key in scraped_metadata.keys() if key in websites]

    for provider_name, url in websites.items():
        if provider_name in successful_scrapes:
            continue

        try:
            logger.info(f"Scraping {provider_name}: {url}")
            scrape_result = app.scrape(url, formats=formats).model_dump()
            metadata = {}
            metadata["provider_name"] = provider_name
            metadata["url"] = url
            metadata["domain"] = urlparse(url).netloc
            metadata["scraped_at"] = datetime.now().isoformat()
            metadata["formats"] = formats
            metadata['success'] = True

            content_files = {}
            for fmt in formats:
                content = scrape_result[fmt]
                file_name = f"{provider_name}_{fmt}.txt"
                file_path = os.path.join(SCRAPE_DIR, file_name)

                with open(file_path, "w", encoding="utf-8") as cf:
                    cf.write(content)
                    logger.info(f"Saved {fmt} content for {provider_name} to {file_path}")
                    content_files[fmt] = file_name

            metadata['content_files'] = content_files
            metadata['title'] = scrape_result['metadata']['title']
            metadata['description'] = scrape_result['metadata']['description']
            successful_scrapes.append(provider_name)

            scraped_metadata[provider_name] = metadata
        except Exception as e:
            logger.warning(f"Unable to scrape {provider_name}: {e}")

    # Write the entire scraped_metadata dictionary back to the scraped_metadata.json file
    with open(metadata_file, "w", encoding="utf-8") as mf_out:
        json.dump(scraped_metadata, mf_out, indent=4, ensure_ascii=False)

    return successful_scrapes


@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.
    
    Args:
        identifier: The provider name, full URL, or domain to look for
        
    Returns:
        Formatted JSON string with the scraped information
    """
    
    logger.info(f"Extracting information for identifier: {identifier}")
    logger.info(f"Files in {SCRAPE_DIR}: {os.listdir(SCRAPE_DIR)}")

    metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    logger.info(f"Checking metadata file: {metadata_file}")

    # continue your response here ...
    try:
        if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
            with open(metadata_file, "r", encoding="utf-8") as mf:
                scraped_metadata = json.load(mf) or {}
        else:
            scraped_metadata = {}
    except Exception as e:
        logger.warning(f"Unable to load metadata: {e}")
        scraped_metadata = {}

    for provider_name, metadata in scraped_metadata.items():
        if identifier == provider_name or identifier == metadata.get("url", "") or identifier == metadata.get("domain", ""):
            logger.info(f"Found matching metadata for {identifier}")
            result = metadata.copy()

            if 'content_files' in metadata:
                result['content'] = {}

                for format_type, filename in metadata['content_files'].items():
                    file_path = os.path.join(SCRAPE_DIR, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as cf:
                            content = cf.read()
                            result['content'][format_type] = content
                    except Exception as e:
                        logger.warning(f"Unable to read content file {filename}: {e}")

            return json.dumps(result, indent=2)
    else:
        return f"There's no saved information related to identifier '{identifier}'."


if __name__ == "__main__":
    mcp.run(transport="stdio")
