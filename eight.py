import os
import time
import threading
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Literal, Any
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
import re
from cachetools import TTLCache
from contextlib import asynccontextmanager

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration with environment variables
class Config:
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_API")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "YOUR_API")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_ENDPOINT")
    UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 3600))  # 1 hour
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 5))
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", 500))
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3500))  # Slightly less than update interval

# Enhanced AI/ML keywords with categories
AI_KEYWORDS = {
    'general': {'ai', 'artificial intelligence', 'machine learning', 'ml'},
    'models': {'llm', 'gpt', 'chatgpt', 'transformer', 'diffusion', 'llama'},
    'tech': {'pytorch', 'tensorflow', 'huggingface', 'langchain', 'openai'},
    'domains': {'nlp', 'computer vision', 'generative ai', 'reinforcement learning'}
}

# Cache setup
cache = TTLCache(maxsize=Config.CACHE_SIZE, ttl=Config.CACHE_TTL)

# Pydantic models for API responses
class ContentItem(BaseModel):
    id: str
    title: str
    thumbnail: Optional[str] = None
    summary: str
    source: str
    content_type: Literal["tool", "model", "paper", "video", "article"]
    date: datetime
    link: str
    popularity_score: int
    tags: List[str] = []

class FeedResponse(BaseModel):
    data: List[ContentItem]
    next_cursor: Optional[str] = None
    has_more: bool

class ContentSource(BaseModel):
    name: str
    enabled: bool = True
    priority: int = Field(ge=1, le=10, default=5)

class ContentConfig(BaseModel):
    sources: List[ContentSource]
    default_limit: int = Field(20, ge=1, le=100)
    max_limit: int = 100
    update_interval: int = 3600

@dataclass
class ContentFetcherConfig:
    news_count: int = 20
    github_count: int = 20
    youtube_count: int = 20
    arxiv_count: int = 20
    hf_count: int = 20
    min_stars: int = 100
    min_downloads: int = 1000
    min_views: int = 10000

class ContentTracker:
    def __init__(self):
        self.last_updated = None
        self.cached_data = {
            'news': [],
            'github_repos': [],
            'youtube_videos': [],
            'arxiv_papers': [],
            'hf_models': []
        }
        self.config = ContentFetcherConfig()
        self.lock = threading.Lock()
        self.is_refreshing = False
        self.content_config = ContentConfig(
            sources=[
                ContentSource(name="GitHub", priority=8),
                ContentSource(name="YouTube", priority=7),
                ContentSource(name="ArXiv", priority=6),
                ContentSource(name="HuggingFace", priority=9),
                ContentSource(name="News", priority=5)
            ]
        )

    def refresh_data(self, force=False):
        with self.lock:
            if self.is_refreshing and not force:
                logger.info("Refresh already in progress")
                return False

            self.is_refreshing = True
            try:
                logger.info("Starting comprehensive data refresh...")
                start_time = time.time()
                
                tasks = {
                    'news': self.fetch_ai_news,
                    'github_repos': self.fetch_github_trending,
                    'youtube_videos': self.fetch_youtube_videos,
                    'arxiv_papers': self.fetch_arxiv_papers,
                    'hf_models': self.fetch_huggingface_trends
                }

                results = {}
                with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                    future_to_key = {executor.submit(task): key for key, task in tasks.items()}
                    for future in as_completed(future_to_key):
                        key = future_to_key[future]
                        try:
                            results[key] = future.result()
                            logger.info(f"Completed fetching {key} with {len(results[key])} items")
                        except Exception as e:
                            logger.error(f"Error fetching {key}: {str(e)}", exc_info=True)
                            results[key] = []

                self.cached_data.update(results)
                self.last_updated = datetime.now(timezone.utc)
                
                # Normalize all data to ContentItem format
                for key in self.cached_data:
                    self.cached_data[key] = [self.normalize_item(item, key) for item in self.cached_data[key]]
                
                logger.info(f"Refresh completed in {time.time() - start_time:.2f} seconds")
                return True
            except Exception as e:
                logger.error(f"Refresh failed: {str(e)}", exc_info=True)
                return False
            finally:
                self.is_refreshing = False

    def normalize_item(self, item: Dict, source_type: str) -> ContentItem:
        """Convert raw data from different sources to standardized ContentItem"""
        # Generate a unique ID if not present
        item_id = item.get('id')
        if not item_id:
            # Create a hashable version of the item for ID generation
            hashable_item = {}
            for k, v in item.items():
                if isinstance(v, (list, dict)):
                    hashable_item[k] = str(v)
                else:
                    hashable_item[k] = v
            item_id = str(hash(frozenset(hashable_item.items())))
        
        base_data = {
            "id": f"{source_type[:3]}_{item_id}",
            "title": item.get('title', 'Untitled'),
            "thumbnail": item.get('thumbnail'),
            "summary": item.get('summary', ''),
            "source": source_type,
            "date": item.get('date', datetime.now(timezone.utc)),
            "link": item.get('url', item.get('link', '#')),
            "popularity_score": item.get('popularity_score', 0),
            "tags": item.get('tags', [])
        }
        
        # Map source_type to content_type
        type_mapping = {
            'news': 'article',
            'github_repos': 'tool',
            'youtube_videos': 'video',
            'arxiv_papers': 'paper',
            'hf_models': 'model'
        }
        
        base_data["content_type"] = type_mapping.get(source_type, 'article')
        return ContentItem(**base_data)

    def fetch_ai_news(self) -> List[Dict]:
        """Fetch AI-related news from NewsAPI with enhanced filtering"""
        try:
            # Join keywords with OR for the query
            query = ' OR '.join(AI_KEYWORDS['general'])
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize=100&language=en&apiKey={Config.NEWS_API_KEY}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            articles = [
                self.process_news_article(article) 
                for article in response.json().get('articles', [])
                if self.is_ai_related(article)
            ][:self.config.news_count]
            
            return articles
        except Exception as e:
            logger.error(f"NewsAPI error: {str(e)}")
            return []

    def process_news_article(self, article: Dict) -> Dict:
        """Extract and process relevant fields from news article"""
        content = article.get('content', '')
        return {
            'title': article.get('title', 'Untitled'),
            'summary': content[:200] + '...' if content else 'No summary available',
            'url': article.get('url', '#'),
            'date': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc) if article.get('publishedAt') else datetime.now(timezone.utc),
            'thumbnail': article.get('urlToImage'),
            'popularity_score': 0,  # NewsAPI doesn't provide popularity metrics
            'tags': self.extract_tags(article),
            'source': article.get('source', {}).get('name', 'Unknown')
        }

    def fetch_github_trending(self) -> List[Dict]:
        """Fetch trending GitHub repos with AI focus"""
        try:
            repos = []
            for page in range(1, 3):  # Check first 2 pages
                if len(repos) >= self.config.github_count:
                    break
                    
                url = f"https://api.github.com/search/repositories?q=stars:>={self.config.min_stars}+topic:ai&sort=stars&order=desc&page={page}&per_page=50"
                headers = {"Authorization": f"token {Config.GITHUB_TOKEN}"} if Config.GITHUB_TOKEN else {}
                
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                
                new_repos = [
                    self.process_github_repo(repo) 
                    for repo in response.json().get('items', [])
                    if self.is_ai_related_repo(repo)
                ]
                repos.extend(new_repos)
            
            return repos[:self.config.github_count]
        except Exception as e:
            logger.error(f"GitHub API error: {str(e)}")
            return []

    def process_github_repo(self, repo: Dict) -> Dict:
        """Process GitHub repository data"""
        return {
            'id': str(repo['id']),
            'title': repo['name'],
            'summary': repo['description'] or 'No description',
            'url': repo['html_url'],
            'date': datetime.strptime(repo['created_at'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc),
            'thumbnail': repo['owner']['avatar_url'],
            'popularity_score': repo['stargazers_count'],
            'tags': repo.get('topics', []),
            'source': 'GitHub'
        }

    def fetch_youtube_videos(self) -> List[Dict]:
        """Fetch AI-related YouTube videos"""
        try:
            # Join keywords with + for the query
            query = '+'.join(AI_KEYWORDS['general'])
            url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=50&q={query}&type=video&order=viewCount&key={Config.YOUTUBE_API_KEY}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            videos = [
                self.process_youtube_video(item) 
                for item in response.json().get('items', [])
                if self.is_ai_related_video(item)
            ][:self.config.youtube_count]
            
            return videos
        except Exception as e:
            logger.error(f"YouTube API error: {str(e)}")
            return []

    def process_youtube_video(self, video: Dict) -> Dict:
        """Process YouTube video data"""
        snippet = video['snippet']
        return {
            'id': video['id']['videoId'],
            'title': snippet['title'],
            'summary': snippet['description'][:200] + '...' if snippet.get('description') else 'No description available',
            'url': f"https://youtube.com/watch?v={video['id']['videoId']}",
            'date': datetime.strptime(snippet['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc),
            'thumbnail': snippet['thumbnails']['high']['url'] if 'high' in snippet.get('thumbnails', {}) else None,
            'popularity_score': 0,  # Would need another API call to get views
            'tags': snippet.get('tags', []),
            'source': 'YouTube'
        }

    def fetch_arxiv_papers(self) -> List[Dict]:
        """Fetch recent AI papers from arXiv"""
        try:
            url = "http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG&sortBy=submittedDate&sortOrder=descending&max_results=50"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'xml')
            entries = soup.find_all('entry')
            
            papers = [
                self.process_arxiv_paper(entry)
                for entry in entries
            ][:self.config.arxiv_count]
            
            return papers
        except Exception as e:
            logger.error(f"arXiv API error: {str(e)}")
            return []

    def process_arxiv_paper(self, paper) -> Dict:
        """Process arXiv paper data"""
        try:
            paper_id = paper.id.text.split('/')[-1]
            published_date = datetime.strptime(paper.published.text, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            summary = paper.summary.text.strip()
            
            return {
                'id': paper_id,
                'title': paper.title.text.strip(),
                'summary': summary[:200] + '...' if len(summary) > 200 else summary,
                'url': paper.id.text,
                'date': published_date,
                'thumbnail': None,  # arXiv doesn't provide thumbnails
                'popularity_score': 0,  # arXiv doesn't provide metrics
                'tags': [cat['term'] for cat in paper.find_all('category')],
                'source': 'ArXiv'
            }
        except Exception as e:
            logger.error(f"Error processing arXiv paper: {str(e)}")
            return {
                'id': 'unknown',
                'title': 'Error processing paper',
                'summary': 'Unable to process this paper',
                'url': '#',
                'date': datetime.now(timezone.utc),
                'thumbnail': None,
                'popularity_score': 0,
                'tags': [],
                'source': 'ArXiv'
            }

    def fetch_huggingface_trends(self) -> List[Dict]:
        """Fetch trending models from HuggingFace"""
        try:
            url = "https://huggingface.co/api/models?sort=downloads&direction=-1"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            models = [
                self.process_hf_model(model)
                for model in response.json()
                if model.get('downloads', 0) >= self.config.min_downloads and self.is_ai_related_model(model)
            ][:self.config.hf_count]
            
            return models
        except Exception as e:
            logger.error(f"HuggingFace API error: {str(e)}")
            return []

    def process_hf_model(self, model: Dict) -> Dict:
        """Process HuggingFace model data"""
        description = model.get('description', 'No description')
        return {
            'id': model['modelId'],
            'title': model['modelId'].split('/')[-1],
            'summary': description[:200] + '...' if len(description) > 200 else description,
            'url': f"https://huggingface.co/{model['modelId']}",
            'date': datetime.strptime(model.get('lastModified', '1970-01-01T00:00:00.000Z'), '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc),
            'thumbnail': None,  # HF doesn't provide thumbnails
            'popularity_score': model.get('downloads', 0),
            'tags': model.get('tags', []),
            'source': 'HuggingFace'
        }

    # Helper methods for content filtering
    def is_ai_related(self, content: Dict) -> bool:
        """Check if content is AI-related based on multiple fields"""
        text = ' '.join([
            content.get('title', '').lower(),
            content.get('description', '').lower(),
            content.get('content', '').lower(),
            content.get('summary', '').lower()
        ])
        return any(re.search(rf'\b{keyword}\b', text) 
               for category in AI_KEYWORDS.values() 
               for keyword in category)

    def is_ai_related_repo(self, repo: Dict) -> bool:
        """Check if GitHub repo is AI-related"""
        text = ' '.join([
            repo.get('name', '').lower(),
            repo.get('description', '').lower() if repo.get('description') else '',
            ' '.join(repo.get('topics', []))
        ])
        return any(re.search(rf'\b{keyword}\b', text) 
               for category in AI_KEYWORDS.values() 
               for keyword in category)

    def is_ai_related_video(self, video: Dict) -> bool:
        """Check if YouTube video is AI-related"""
        snippet = video.get('snippet', {})
        text = ' '.join([
            snippet.get('title', '').lower(),
            snippet.get('description', '').lower() if snippet.get('description') else '',
            ' '.join(snippet.get('tags', []))
        ])
        return any(re.search(rf'\b{keyword}\b', text) 
               for category in AI_KEYWORDS.values() 
               for keyword in category)

    def is_ai_related_model(self, model: Dict) -> bool:
        """Check if HF model is AI-related"""
        text = ' '.join([
            model.get('modelId', '').lower(),
            model.get('pipeline_tag', '').lower() if model.get('pipeline_tag') else '',
            model.get('description', '').lower() if model.get('description') else '',
            ' '.join(model.get('tags', []))
        ])
        return any(re.search(rf'\b{keyword}\b', text) 
               for category in AI_KEYWORDS.values() 
               for keyword in category)

    def extract_tags(self, content: Dict) -> List[str]:
        """Extract relevant tags from content"""
        text = ' '.join([
            content.get('title', '').lower(),
            content.get('description', '').lower() if content.get('description') else '',
            content.get('content', '').lower() if content.get('content') else ''
        ])
        return [keyword for category in AI_KEYWORDS.values() 
                for keyword in category 
                if re.search(rf'\b{keyword}\b', text)]

    # Content fetching methods
    def get_all_content(self) -> List[ContentItem]:
        """Get all content items combined"""
        if self.needs_refresh():
            threading.Thread(target=self.refresh_data).start()
        
        all_content = []
        for source in self.cached_data.values():
            all_content.extend(source)
        
        return all_content

    def get_filtered_content(
        self,
        content_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        min_score: int = 0
    ) -> List[ContentItem]:
        """Get filtered content based on parameters"""
        all_content = self.get_all_content()
        
        filtered = []
        for item in all_content:
            if content_types and item.content_type not in content_types:
                continue
            if sources and item.source not in sources:
                continue
            if min_date and item.date < min_date:
                continue
            if max_date and item.date > max_date:
                continue
            if item.popularity_score < min_score:
                continue
            filtered.append(item)
        
        return filtered

    def needs_refresh(self) -> bool:
        """Check if data needs refreshing"""
        if not self.last_updated:
            return True
        return (datetime.now(timezone.utc) - self.last_updated).total_seconds() > Config.UPDATE_INTERVAL

# FastAPI App Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize data on startup"""
    threading.Thread(target=tracker.refresh_data, kwargs={'force': True}).start()
    yield

app = FastAPI(
    title="AI Content Feed API",
    description="API for infinite-scroll feed of AI-related content from multiple sources",
    version="1.0.0",
    lifespan=lifespan
)

tracker = ContentTracker()

# API Endpoints
@app.get("/feed", response_model=FeedResponse)
async def get_feed(
    cursor: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    content_types: Optional[List[str]] = Query(None),
    sources: Optional[List[str]] = Query(None),
    sort_by: str = Query("latest", pattern="^(latest|popular|oldest)$"),
    min_score: int = Query(0, ge=0)
):
    """
    Get paginated feed of AI content with optional filtering and sorting.
    
    Parameters:
    - cursor: Pagination cursor (omit for first page)
    - limit: Number of items per page (1-100)
    - content_types: Filter by content types (tool, model, paper, video, article)
    - sources: Filter by sources (GitHub, YouTube, ArXiv, HuggingFace, News)
    - sort_by: Sorting method (latest, popular, oldest)
    - min_score: Minimum popularity score
    """
    try:
        # Get filtered content
        filtered = tracker.get_filtered_content(
            content_types=content_types,
            sources=sources,
            min_score=min_score
        )
        
        # Apply sorting
        if sort_by == "latest":
            filtered.sort(key=lambda x: x.date, reverse=True)
        elif sort_by == "oldest":
            filtered.sort(key=lambda x: x.date)
        elif sort_by == "popular":
            filtered.sort(key=lambda x: x.popularity_score, reverse=True)
        
        # Handle pagination
        start_idx = 0
        if cursor:
            try:
                cursor_time, cursor_id = cursor.split("_")
                cursor_time = datetime.fromisoformat(cursor_time)
                
                # Find position after cursor
                for i, item in enumerate(filtered):
                    if (sort_by == "latest" and item.date < cursor_time) or \
                       (sort_by == "oldest" and item.date > cursor_time) or \
                       (sort_by == "popular" and item.popularity_score < int(cursor_id)):
                        start_idx = i
                        break
                    if item.id == cursor_id:
                        start_idx = i + 1
                        break
            except Exception as e:
                logger.error(f"Cursor error: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid cursor format")
        
        # Get paginated items
        end_idx = start_idx + limit
        page_items = filtered[start_idx:end_idx]
        has_more = len(filtered) > end_idx
        
        # Create next cursor
        next_cursor = None
        if has_more and page_items:
            last_item = page_items[-1]
            if sort_by == "popular":
                next_cursor = f"{last_item.date.isoformat()}_{last_item.popularity_score}"
            else:
                next_cursor = f"{last_item.date.isoformat()}_{last_item.id}"
        
        return FeedResponse(
            data=page_items,
            next_cursor=next_cursor,
            has_more=has_more
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feed error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/refresh")
async def trigger_refresh(force: bool = False):
    """
    Manually trigger a content refresh.
    """
    try:
        result = {
            "status": "refresh_started",
            "last_updated": tracker.last_updated.isoformat() if tracker.last_updated else None
        }
        threading.Thread(target=tracker.refresh_data, kwargs={'force': force}).start()
        return result
    except Exception as e:
        logger.error(f"Refresh error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "last_updated": tracker.last_updated.isoformat() if tracker.last_updated else None,
        "sources": [source.name for source in tracker.content_config.sources if source.enabled],
        "content_counts": {source: len(items) for source, items in tracker.cached_data.items()}
    }

@app.get("/sources")
async def get_sources():
    """
    Get available content sources.
    """
    return {
        "sources": [
            {
                "name": source.name,
                "enabled": source.enabled,
                "priority": source.priority
            } for source in tracker.content_config.sources
        ]
    }

@app.get("/content_types")
async def get_content_types():
    """
    Get available content types.
    """
    return {
        "content_types": ["tool", "model", "paper", "video", "article"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
