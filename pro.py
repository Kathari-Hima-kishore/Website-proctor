import os
from dotenv import load_dotenv
load_dotenv()

import asyncio
import numpy as np
import io
import base64
from playwright.async_api import async_playwright
import requests
import azure.cognitiveservices.vision.computervision as computervision
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import json
import time
from datetime import datetime
import logging
import threading
import webbrowser
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import random

# --- Begin AzureConfig class (inlined from azure_config.py) ---
from dotenv import load_dotenv
import requests
from azure.core.exceptions import AzureError

class AzureConfig:
    def __init__(self):
        load_dotenv()
        self.cv_key = os.getenv('AZURE_CV_KEY', 'your_computer_vision_key')
        self.cv_endpoint = os.getenv('AZURE_CV_ENDPOINT', 'your_computer_vision_endpoint')
        self.text_key = os.getenv('AZURE_TEXT_KEY', 'your_text_analytics_key')
        self.text_endpoint = os.getenv('AZURE_TEXT_ENDPOINT', 'your_text_analytics_endpoint')
        self.load_from_file()
        self.cv_client = None
        self.text_client = None
        self.connection_status = {
            'computer_vision': False,
            'text_analytics': False,
            'last_tested': None,
            'errors': []
        }
    def load_from_file(self):
        config_file = 'azure_credentials.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.cv_key = config.get('computer_vision_key', self.cv_key)
                    self.cv_endpoint = config.get('computer_vision_endpoint', self.cv_endpoint)
                    self.text_key = config.get('text_analytics_key', self.text_key)
                    self.text_endpoint = config.get('text_analytics_endpoint', self.text_endpoint)
            except Exception as e:
                print(f"Error loading config file: {e}")
    def save_to_file(self):
        config = {
            'computer_vision_key': self.cv_key,
            'computer_vision_endpoint': self.cv_endpoint,
            'text_analytics_key': self.text_key,
            'text_analytics_endpoint': self.text_endpoint,
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open('azure_credentials.json', 'w') as f:
                json.dump(config, f, indent=2)
            print("✅ Azure credentials saved to azure_credentials.json")
        except Exception as e:
            print(f"❌ Error saving config file: {e}")    
    def test_computer_vision_connection(self):
        try:
            if self.cv_key == 'your_computer_vision_key' or self.cv_endpoint == 'your_computer_vision_endpoint':
                raise Exception("Please configure your Computer Vision credentials")
            
            # Check if key looks valid (Azure keys are typically 32 characters)
            if len(self.cv_key) < 30:
                raise Exception("Computer Vision API key appears to be incomplete or truncated")
                
            self.cv_client = computervision.ComputerVisionClient(
                self.cv_endpoint, 
                CognitiveServicesCredentials(self.cv_key)
            )
            test_image = Image.new('RGB', (100, 100), color='blue')
            img_byte_arr = BytesIO()
            test_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            analysis = self.cv_client.analyze_image_in_stream(
                img_byte_arr,
                visual_features=[VisualFeatureTypes.description, VisualFeatureTypes.color]
            )
            self.connection_status['computer_vision'] = True
            print("✅ Computer Vision API connection successful!")
            return True
        except Exception as e:
            self.connection_status['computer_vision'] = False
            error_msg = str(e)
            if "401" in error_msg:
                error_msg = "Invalid API key or expired subscription. Please check your Computer Vision credentials."
            elif "403" in error_msg:
                error_msg = "Access forbidden. Please check your Computer Vision subscription and billing status."
            elif "truncated" in error_msg:
                error_msg = "Computer Vision API key appears incomplete. Please check your .env file."
            self.connection_status['errors'].append(f"Computer Vision: {error_msg}")
            print(f"❌ Computer Vision API connection failed: {error_msg}")
            return False
    def test_text_analytics_connection(self):
        try:
            if self.text_key == 'your_text_analytics_key' or self.text_endpoint == 'your_text_analytics_endpoint':
                raise Exception("Please configure your Text Analytics credentials")
            self.text_client = TextAnalyticsClient(
                endpoint=self.text_endpoint,
                credential=AzureKeyCredential(self.text_key)
            ) 
            test_documents = ["This is a test message for Azure Text Analytics."]
            response = self.text_client.analyze_sentiment(documents=test_documents)
            if response and len(response) > 0:
                self.connection_status['text_analytics'] = True
                print("✅ Text Analytics API connection successful!")
                return True
            else:
                raise Exception("No response from Text Analytics API")
        except Exception as e:
            self.connection_status['text_analytics'] = False
            self.connection_status['errors'].append(f"Text Analytics: {str(e)}")
            print(f"❌ Text Analytics API connection failed: {e}")
            return False
    def test_all_connections(self):
        print("\n🔍 Testing Azure Connections...")
        print("=" * 40)
        self.connection_status['errors'] = []
        self.connection_status['last_tested'] = datetime.now().isoformat()
        cv_success = self.test_computer_vision_connection()
        text_success = self.test_text_analytics_connection()
        print("\n📊 Connection Status Summary:")
        print(f"Computer Vision: {'✅ Connected' if cv_success else '❌ Failed'}")
        print(f"Text Analytics: {'✅ Connected' if text_success else '❌ Failed'}")
        if cv_success and text_success:
            print("\n🎉 All Azure services are connected and ready!")
        else:
            print("\n⚠️  Some Azure services failed to connect.")
            print("Please check your credentials and try again.")
        # Return a detailed status object for the frontend
        return {
            "computer_vision": {
                "success": cv_success,
                "message": "Connected" if cv_success else "Failed"
            },
            "text_analytics": {
                "success": text_success,
                "message": "Connected" if text_success else "Failed"
            },
            "errors": self.connection_status.get("errors", [])
        }
    def get_mock_clients(self):
        print("🔧 Using mock Azure clients for development")
        class MockCVClient:
            def analyze_image_in_stream(self, image_stream, visual_features=None):
                class MockAnalysis:
                    def __init__(self):
                        self.description = type('obj', (object,), {'captions': [type('obj', (object,), {'text': 'Mock analysis', 'confidence': 0.9})]})()
                        self.color = type('obj', (object,), {'is_bw_img': False, 'accent_color': 'blue'})()
                        self.tags = [type('obj', (object,), {'name': 'webpage', 'confidence': 0.95})()]
                return MockAnalysis()
        class MockTextClient:
            def analyze_sentiment(self, documents=None):
                class MockSentiment:
                    def __init__(self, text):
                        self.text = text
                        self.sentiment = 'neutral'
                return [MockSentiment(doc) for doc in documents]
        return MockCVClient(), MockTextClient()
    def get_clients(self):
        if self.connection_status['computer_vision'] and self.connection_status['text_analytics']:
            return self.cv_client, self.text_client
        else:
            return self.get_mock_clients()
# --- End AzureConfig class ---

# Setup logging
logging.basicConfig(filename='website_proctor.log', level=logging.INFO)

# Initialize Azure configuration
azure_config = AzureConfig()

# Flask and SocketIO setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'website_proctor_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
proctor_instance = None
is_monitoring = False

class WebsiteProctor:
    def __init__(self):
        self.page = None
        self.browser = None
        self.playwright = None
        
        # Initialize Azure clients using the configuration
        global azure_config
        self.azure_config = azure_config  # Add missing azure_config reference
        self.cv_client, self.text_client = azure_config.get_clients()
        self.azure_connected = azure_config.connection_status['computer_vision'] and azure_config.connection_status['text_analytics']
            
        # Fix issues structure - should be a dict with categories
        self.issues = {
            'security': [],
            'performance': [],
            'accessibility': [],
            'ui': []
        }
        self.suggestions = []
        self.is_running = False
        self.metrics = {
            "load_time": 0,
            "performance_score": 85,            "accessibility_score": 90,
            "security_status": "Good",
            "uptime": "100%",
            "azure_status": "Connected" if self.azure_connected else "Mock/Disconnected"
        }
        self.activity_log = []

    def add_issue(self, category, severity, message):
        """Add an issue to the specified category"""
        issue = {
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        if category in self.issues:
            self.issues[category].append(issue)
        else:
            self.issues[category] = [issue]
        
        # Debug logging to verify issues are being added correctly
        self.log_activity(f"Added {severity} {category} issue: {message}", "debug")

    def log_activity(self, message, level="info"):
        """Log activity with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.append({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })
        
        # Keep only last 50 entries
        if len(self.activity_log) > 50:
            self.activity_log = self.activity_log[-50:]
          # Emit to dashboard
        socketio.emit('activity_update', {
            "timestamp": timestamp,
            "message": message,
            "level": level
        })

    async def start_browser(self, url):
        """Start Playwright browser and load website for analysis"""
        try:
            # Validate URL first
            if not url.startswith(('http://', 'https://')):
                # Try to validate if it's a real website
                import requests
                test_url = f"https://{url}" if not url.startswith(('http://', 'https://')) else url
                try:
                    response = requests.head(test_url, timeout=10, allow_redirects=True)
                    if response.status_code >= 400:
                        raise Exception(f"Website returned {response.status_code} error")
                    url = test_url
                except requests.exceptions.RequestException as e:
                    socketio.emit('website_error', {
                        'error': f"Cannot access website: {str(e)}. Please check if the URL is correct and the website is accessible."
                    })
                    return False
            
            # Use only Playwright for analysis - no Chrome popup
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=False,  # Set to False so we can see what's happening
                args=['--window-size=1200,800']
            )
            context = await self.browser.new_context(
                viewport={'width': 1200, 'height': 800}
            )
            self.page = await context.new_page()
            
            # Navigate to the website
            await self.page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Take screenshot and send to dashboard
            screenshot_bytes = await self.page.screenshot(full_page=True)
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            # Send screenshot to dashboard iframe
            socketio.emit('website_screenshot', {
                'url': url,
                'screenshot': f"data:image/png;base64,{screenshot_base64}",
                'timestamp': datetime.now().isoformat()
            })
            
            self.log_activity(f"Successfully loaded website: {url}")
            socketio.emit('current_action', {
                'action': f'Website loaded successfully: {url}', 
                'phase': 1
            })
            return True
            
        except Exception as e:
            error_msg = f"Failed to load website: {str(e)}"
            self.log_activity(error_msg, "error")
            socketio.emit('website_error', {'error': error_msg})
            return False

    async def capture_screenshot(self):
        """Capture screenshot using Playwright for analysis"""
        try:
            if self.page:
                # Take screenshot with Playwright
                screenshot_bytes = await self.page.screenshot(full_page=True)
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                
                # Send updated screenshot to dashboard
                socketio.emit('website_screenshot', {
                    'screenshot': f"data:image/png;base64,{screenshot_base64}",
                    'timestamp': datetime.now().isoformat()
                })
                
                # Save screenshot for Azure analysis
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}.png"
                with open(screenshot_path, 'wb') as f:
                    f.write(screenshot_bytes)
                
                return screenshot_path
            else:
                raise Exception("No browser page available")
        except Exception as e:
            self.log_activity(f"Screenshot capture failed: {str(e)}", "error")
            return None

    def analyze_image(self, image_path):
        """Analyze image using Azure Computer Vision or mock analysis"""
        try:
            if self.azure_connected and self.cv_client:
                with open(image_path, "rb") as image_stream:
                    analysis = self.cv_client.analyze_image_in_stream(
                        image_stream, visual_features=[
                            VisualFeatureTypes.description,
                            VisualFeatureTypes.tags,
                            VisualFeatureTypes.color
                        ]
                    )
                self.log_activity("Azure Computer Vision analysis completed")
                return analysis
            else:                # Use mock analysis
                self.log_activity("Using mock image analysis (Azure not connected)")
                return self.analyze_image_mock(image_path)
        except Exception as e:
            self.log_activity(f"Image analysis failed: {str(e)}", "error")
            return self.analyze_image_mock(image_path)
    
    def analyze_image_mock(self, image_path):
        """Mock image analysis for development/fallback"""
        return {
            "description": {"captions": [{"text": "Web page screenshot", "confidence": 0.9}]},
            "color": {"is_bw_img": False, "accent_color": "blue"},
            "tags": [{"name": "webpage", "confidence": 0.95}]
        }

    async def analyze_accessibility(self):
        """Analyze accessibility issues using Playwright"""
        try:
            accessibility_issues = await self.page.evaluate("""
                () => {
                    const issues = [];
                    
                    // Check for missing alt text
                    document.querySelectorAll('img').forEach(img => {
                        if (!img.alt || img.alt.trim() === '') {
                            issues.push('Image missing alt text');
                        }
                    });
                    
                    // Check for links without accessible names
                    document.querySelectorAll('a').forEach(a => {
                        if (!a.getAttribute('aria-label') && 
                            !a.textContent.trim() && 
                            !a.querySelector('img[alt]')) {
                            issues.push('Link missing accessible name');
                        }
                    });
                    
                    // Check for form inputs without labels
                    document.querySelectorAll('input[type="text"], input[type="email"], textarea').forEach(input => {
                        const id = input.id;
                        if (!id || !document.querySelector(`label[for="${id}"]`)) {
                            issues.push('Form input missing label');
                        }
                    });
                    
                    // Check for headings structure
                    const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
                    if (headings.length === 0) {
                        issues.push('No heading structure found');
                    }
                    
                    return issues;
                }
            """)
            
            # Add accessibility issues
            for issue in accessibility_issues:
                self.add_issue("accessibility", "minor", issue)
                
            self.log_activity(f"Accessibility analysis completed - found {len(accessibility_issues)} issues")
            return accessibility_issues
            
        except Exception as e:
            self.log_activity(f"Accessibility analysis failed: {str(e)}", "error")
            return []

    async def analyze_network_security(self):
        """Analyze network security and integrity using Playwright"""
        try:
            # Check HTTPS usage
            current_url = self.page.url
            if not current_url.startswith("https"):
                self.add_issue("security", "major", "Insecure connection (HTTP instead of HTTPS)")
                self.metrics["security_status"] = "Poor"
            else:
                self.metrics["security_status"] = "Good"
            
            # Check for mixed content and security headers
            security_info = await self.page.evaluate("""
                () => {
                    const issues = [];
                    const info = {};
                    
                    // Check for mixed content
                    const mixedContent = Array.from(document.querySelectorAll('img, script, link')).filter(el => {
                        const src = el.src || el.href;
                        return src && src.startsWith('http://') && window.location.protocol === 'https:';
                    });
                    
                    if (mixedContent.length > 0) {
                        issues.push(`Mixed content detected: ${mixedContent.length} insecure resources`);
                    }
                    
                    // Check for password fields over HTTP
                    const passwordFields = document.querySelectorAll('input[type="password"]');
                    if (passwordFields.length > 0 && window.location.protocol === 'http:') {
                        issues.push('Password fields detected over insecure HTTP connection');
                    }
                    
                    // Check for external scripts
                    const externalScripts = Array.from(document.querySelectorAll('script[src]')).filter(script => {
                        const src = script.src;
                        return src && !src.includes(window.location.hostname);
                    });
                    
                    info.externalScripts = externalScripts.length;
                    info.passwordFields = passwordFields.length;
                    info.mixedContent = mixedContent.length;
                    
                    return { issues, info };
                }
            """)
            
            # Add security issues
            for issue in security_info['issues']:
                self.add_issue("security", "major", issue)
            
            # Update metrics
            self.metrics['external_scripts'] = security_info['info']['externalScripts']
            self.metrics['password_fields'] = security_info['info']['passwordFields']
            
            self.log_activity(f"Security analysis completed - found {len(security_info['issues'])} issues")
            
            # Emit security metrics
            socketio.emit('metrics_update', self.metrics)
            
        except Exception as e:
            self.log_activity(f"Network security analysis failed: {str(e)}", "error")

    async def analyze_performance(self):
        """Analyze website performance using Playwright"""
        try:
            # Get performance metrics
            performance_metrics = await self.page.evaluate("""
                () => {
                    const perfData = performance.getEntriesByType('navigation')[0];
                    const timing = performance.timing;
                    
                    return {
                        loadTime: perfData ? perfData.loadEventEnd - perfData.loadEventStart : 0,
                        domContentLoaded: perfData ? perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart : 0,
                        firstPaint: performance.getEntriesByType('paint').find(entry => entry.name === 'first-paint')?.startTime || 0,
                        resourceCount: performance.getEntriesByType('resource').length,
                        totalTransferSize: performance.getEntriesByType('resource').reduce((total, resource) => total + (resource.transferSize || 0), 0)
                    };
                }
            """)
            
            # Calculate performance score
            load_time = performance_metrics['loadTime']
            performance_score = 100
            
            if load_time > 3000:
                performance_score -= 30
            elif load_time > 1500:
                performance_score -= 15
                
            # Check resource count
            if performance_metrics['resourceCount'] > 50:
                performance_score -= 10
                self.add_issue("performance", "minor", f"High resource count: {performance_metrics['resourceCount']} resources")
            
            # Check transfer size
            transfer_mb = performance_metrics['totalTransferSize'] / (1024 * 1024)
            if transfer_mb > 5:
                performance_score -= 20
                self.add_issue("performance", "major", f"Large page size: {transfer_mb:.1f}MB")
            
            # Update metrics
            self.metrics.update({
                'load_time': int(load_time),
                'performance_score': max(0, performance_score),
                'resource_count': performance_metrics['resourceCount'],
                'page_size_mb': round(transfer_mb, 1)
            })
            
            self.log_activity(f"Performance analysis completed - Score: {performance_score}/100")
            
            # Emit performance metrics
            socketio.emit('metrics_update', self.metrics)
            
        except Exception as e:
            self.log_activity(f"Performance analysis failed: {str(e)}", "error")

    async def analyze_ui_with_cv(self):
        """Analyze UI using Azure Computer Vision"""
        try:
            if not self.page:
                return
              # Take screenshot for Computer Vision analysis
            screenshot_bytes = await self.page.screenshot(full_page=True)
            
            # Use Azure Computer Vision to analyze the UI
            if self.azure_config.cv_client:
                cv_analysis = await self.analyze_image_with_cv(screenshot_bytes)
                
                # Process Computer Vision results for UI insights
                if cv_analysis:
                    # Check for text readability
                    if 'read' in cv_analysis:
                        text_regions = len(cv_analysis['read'].get('readResults', []))
                        if text_regions == 0:
                            self.add_issue("ui", "minor", "No readable text detected by Computer Vision")
                        else:
                            self.log_activity(f"Computer Vision detected {text_regions} text regions")
                    
                    # Check for objects and layout
                    if 'objects' in cv_analysis:
                        objects = cv_analysis['objects']
                        self.log_activity(f"Computer Vision detected {len(objects)} UI objects")
                        
                        # Look for common UI elements
                        ui_elements = [obj for obj in objects if obj.get('object', '').lower() in ['button', 'text', 'image', 'logo']]
                        self.metrics['ui_elements_detected'] = len(ui_elements)
                    
                    # Check for image content
                    if 'description' in cv_analysis:
                        description = cv_analysis['description']
                        if description.get('captions'):
                            main_caption = description['captions'][0]['text']
                            self.log_activity(f"Computer Vision description: {main_caption}")
                              # Use Text Analytics to analyze the description
                            await self.analyze_description_sentiment(main_caption)
            
            self.log_activity("Computer Vision UI analysis completed")
            
        except Exception as e:
            self.log_activity(f"Computer Vision UI analysis failed: {str(e)}", "error")

    async def analyze_image_with_cv(self, image_bytes):
        """Analyze image using Azure Computer Vision"""
        try:
            if not self.azure_config.cv_client:
                return None
            
            # Convert image bytes to stream
            image_stream = io.BytesIO(image_bytes)
            
            # Analyze image with Computer Vision
            analysis = self.azure_config.cv_client.analyze_image_in_stream(
                image_stream,
                visual_features=['Description', 'Objects', 'Tags', 'Categories']
            )
            
            # Also get text from image
            read_response = self.azure_config.cv_client.read_in_stream(image_stream, raw=True)
            read_operation_location = read_response.headers["Operation-Location"]
            operation_id = read_operation_location.split("/")[-1]
            
            # Wait for read operation to complete
            import time
            while True:
                read_result = self.azure_config.cv_client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)
            
            return {
                'description': {
                    'captions': [{'text': analysis.description.captions[0].text, 'confidence': analysis.description.captions[0].confidence}] if analysis.description.captions else [],
                    'tags': [tag.name for tag in analysis.description.tags]
                },                'objects': [{'object': obj.object_property, 'confidence': obj.confidence, 'rectangle': obj.rectangle} for obj in analysis.objects],
                'categories': [{'name': cat.name, 'score': cat.score} for cat in analysis.categories],
                'read': read_result.as_dict() if read_result.status == 'succeeded' else None
            }
            
        except Exception as e:
            self.log_activity(f"Computer Vision analysis error: {str(e)}", "error")
            return None

    async def analyze_description_sentiment(self, text):
        """Analyze text description using Azure Text Analytics"""
        try:
            if not self.azure_config.text_client:
                return
            
            # Analyze sentiment of the UI description
            response = self.azure_config.text_client.analyze_sentiment(documents=[text])
            
            for doc in response:
                if not doc.is_error:
                    sentiment = doc.sentiment
                    confidence = doc.confidence_scores;
                    
                    self.log_activity(f"UI sentiment analysis: {sentiment} (positive: {confidence.positive:.2f}, negative: {confidence.negative:.2f})")
                    
                    # Add issues based on sentiment
                    if sentiment == 'negative' and confidence.negative > 0.7:
                        self.add_issue("ui", "minor", f"Negative UI sentiment detected: {sentiment}")
                    
                    self.metrics['ui_sentiment'] = sentiment
                    self.metrics['ui_sentiment_confidence'] = round(confidence.positive, 2)
            
        except Exception as e:
            self.log_activity(f"Text Analytics sentiment analysis failed: {str(e)}", "error")

    async def generate_final_results(self):
        """Generate final test results and emit to dashboard"""
        try:
            # Compile all issues and suggestions
            all_issues = []
            suggestions = []
            
            # Process issues by category
            for category, issues in self.issues.items():
                for issue in issues:
                    all_issues.append({
                        'category': category,
                        'severity': issue['severity'],
                        'message': issue['message'],
                        'timestamp': issue['timestamp']
                    })
            
            # Generate suggestions based on issues
            if any(issue['category'] == 'security' for issue in all_issues):
                suggestions.append("Implement HTTPS encryption for all pages")
                suggestions.append("Review and secure all external script dependencies")
            
            if any(issue['category'] == 'performance' for issue in all_issues):
                suggestions.append("Optimize image sizes and implement lazy loading")
                suggestions.append("Minimize and compress CSS/JavaScript files")
                suggestions.append("Consider using a Content Delivery Network (CDN)")
            
            if any(issue['category'] == 'accessibility' for issue in all_issues):
                suggestions.append("Add alt text to all images")
                suggestions.append("Ensure proper form labeling")
                suggestions.append("Implement ARIA labels for better screen reader support")
            
            if any(issue['category'] == 'ui' for issue in all_issues):
                suggestions.append("Improve visual contrast and readability")
                suggestions.append("Review UI layout for better user experience")
            
            # Create final results object
            final_results = {
                'issues': all_issues,
                'suggestions': suggestions,
                'metrics': self.metrics,
                'summary': {
                    'total_issues': len(all_issues),
                    'major_issues': len([i for i in all_issues if i['severity'] in ['major', 'critical']]),
                    'minor_issues': len([i for i in all_issues if i['severity'] == 'minor']),
                    'performance_score': self.metrics.get('performance_score', 0),
                    'security_status': self.metrics.get('security_status', 'Unknown'),
                    'analysis_complete': True
                }
            }
            
            # Emit final results
            socketio.emit('test_complete', final_results)
            self.log_activity(f"Analysis complete: {len(all_issues)} issues found, {len(suggestions)} suggestions generated")
            
            return final_results
            
        except Exception as e:
            self.log_activity(f"Failed to generate final results: {str(e)}", "error")
            return None

    async def continuous_monitor(self, url):
        """Main monitoring loop"""
        global is_monitoring
        is_monitoring = True
        
        self.log_activity("Starting website monitoring...")
        socketio.emit('current_action', {
            'action': 'Initializing browser and connecting to website...', 
            'phase': 1
        })
        
        # Start browser
        if not await self.start_browser(url):
            return
        
        # Perform comprehensive analysis
        try:
            # Phase 1: Initial load and screenshot
            socketio.emit('current_action', {
                'action': 'Capturing initial website screenshot...', 
                'phase': 1
            })
            await self.capture_screenshot()
            
            # Phase 2: Network and Security Analysis
            socketio.emit('current_action', {
                'action': 'Analyzing network security and integrity...', 
                'phase': 2
            })
            await self.analyze_network_security()
            
            # Phase 3: Performance Analysis
            socketio.emit('current_action', {
                'action': 'Measuring website performance metrics...', 
                'phase': 3
            })
            await self.analyze_performance()
            
            # Phase 4: Accessibility Analysis
            socketio.emit('current_action', {
                'action': 'Checking accessibility compliance...', 
                'phase': 4
            })
            await self.analyze_accessibility()
            
            # Phase 5: Visual UI Inspection with Computer Vision
            socketio.emit('current_action', {
                'action': 'Performing AI-powered visual UI inspection...', 
                'phase': 5
            })
            await self.analyze_ui_with_cv()
            
            # Phase 6: Generate final results with Text Analytics
            socketio.emit('current_action', {
                'action': 'Generating comprehensive test results...', 
                'phase': 6
            })
            await self.generate_final_results()
            
        except Exception as e:
            self.log_activity(f"Analysis error: {str(e)}", "error")
            socketio.emit('website_error', {'error': str(e)})
        
        # Monitor continuously (optional)
        monitor_count = 0
        while is_monitoring and monitor_count < 3:  # Limit to 3 monitoring cycles
            try:
                await asyncio.sleep(30)  # Wait 30 seconds between checks
                monitor_count += 1
                
                socketio.emit('current_action', {
                    'action': f'Continuous monitoring cycle {monitor_count}/3...', 
                    'phase': 6
                })
                
                # Take updated screenshot
                await self.capture_screenshot()
                
                # Update metrics
                socketio.emit('metrics_update', self.metrics)
                
            except Exception as e:
                self.log_activity(f"Monitoring error: {str(e)}", "error")
                await asyncio.sleep(10)
                
        self.log_activity("Monitoring completed")

    async def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        global is_monitoring
        is_monitoring = False
          # Clean up Playwright browser
        if self.browser:
            try:
                await self.browser.close()
            except:
                pass
                
        if hasattr(self, 'playwright'):
            try:
                await self.playwright.stop()
            except:
                pass
                
        self.log_activity("Browser closed and monitoring stopped")

    def get_dashboard_data(self):
        """Get current dashboard data"""
        # Flatten issues dictionary into a list with description field
        flattened_issues = []
        for category, issues_list in self.issues.items():
            for issue in issues_list:
                flattened_issues.append({
                    'category': category,
                    'severity': issue['severity'],
                    'description': issue['message'],  # Change message to description for dashboard
                    'timestamp': issue['timestamp']
                })
        
        return {
            "metrics": self.metrics,
            "issues": flattened_issues,
            "suggestions": self.suggestions,
            "activity_log": self.activity_log,
            "verdict": self.generate_verdict()
        }

    def generate_verdict(self):
        """Generate final verdict based on analysis"""
        # Count issues across all categories
        major_issues = 0
        minor_issues = 0
        
        for category, issues_list in self.issues.items():
            for issue in issues_list:
                if issue["severity"] == "major":
                    major_issues += 1
                elif issue["severity"] == "minor":
                    minor_issues += 1
        
        if major_issues == 0 and minor_issues == 0:
            return "Excellent! Website is in optimal condition."
        elif major_issues == 0 and minor_issues <= 3:
            return "Good overall state with minor improvements needed."
        elif major_issues <= 2:
            return "Website needs attention for major issues."
        else:
            return "Critical issues detected. Immediate action required."

# Flask Routes
@app.route('/')
def dashboard():
    """Serve the dashboard"""
    return app.send_static_file('dashboard_with_automation.html')

@app.route('/api/azure-test')
def test_azure_connection():
    """Test Azure connection and return status"""
    global azure_config
    
    # Test connections
    test_results = azure_config.test_all_connections()
    
    return jsonify({
        "success": test_results.get("computer_vision", {}).get("success", False) and test_results.get("text_analytics", {}).get("success", False),
        "test_results": test_results,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/azure-status')
def azure_status():
    """Get Azure connection status (alternative endpoint for dashboard)"""
    global azure_config
    
    # Test connections
    test_results = azure_config.test_all_connections()
    
    return jsonify({
        "success": test_results.get("computer_vision", {}).get("success", False) and test_results.get("text_analytics", {}).get("success", False),
        "test_results": test_results,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/azure-config', methods=['GET', 'POST'])
def azure_config_endpoint():
    """Get or update Azure configuration"""
    global azure_config
    
    if request.method == 'POST':
        data = request.get_json()
        
        # Update configuration
        if 'cv_key' in data:
            azure_config.cv_key = data['cv_key']
        if 'cv_endpoint' in data:
            azure_config.cv_endpoint = data['cv_endpoint']
        if 'text_key' in data:
            azure_config.text_key = data['text_key']
        if 'text_endpoint' in data:
            azure_config.text_endpoint = data['text_endpoint']
            
        # Save to file
        azure_config.save_to_file()
        
        # Test new configuration
        success = azure_config.test_all_connections()
        
        return jsonify({
            "message": "Configuration updated",
            "success": success,
            "status": azure_config.connection_status
        })
    else:
        # Return current configuration (masked keys)
        return jsonify({
            "cv_key": azure_config.cv_key[:10] + "..." if len(azure_config.cv_key) > 10 else azure_config.cv_key,
            "cv_endpoint": azure_config.cv_endpoint,
            "text_key": azure_config.text_key[:10] + "..." if len(azure_config.text_key) > 10 else azure_config.text_key,
            "text_endpoint": azure_config.text_endpoint,
            "status": azure_config.connection_status
        })

@app.route('/api/status')
def get_status():
    """Get current monitoring status"""
    global proctor_instance, is_monitoring
    
    if proctor_instance:
        return jsonify({
            "monitoring": is_monitoring,
            "data": proctor_instance.get_dashboard_data()
        })
    else:
        return jsonify({
            "monitoring": False,
            "data": {"metrics": {}, "issues": [], "suggestions": [], "activity_log": []}
        })

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start monitoring a website"""
    global proctor_instance, is_monitoring
    
    data = request.get_json()
    url = data.get('url', 'https://example.com')
    
    if is_monitoring:
        return jsonify({"error": "Already monitoring"}), 400
    
    proctor_instance = WebsiteProctor()
    
    # Start monitoring in background thread
    def run_monitor():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(proctor_instance.continuous_monitor(url))
        loop.close()
    
    monitor_thread = threading.Thread(target=run_monitor, daemon=True)
    monitor_thread.start()
    
    return jsonify({"message": f"Started monitoring {url}", "status": "success"})

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop monitoring"""
    global proctor_instance, is_monitoring
    
    if not is_monitoring:
        return jsonify({"error": "Not monitoring"}), 400
    
    # Stop monitoring
    async def stop_async():
        if proctor_instance:
            await proctor_instance.stop_monitoring()
    
    # Run in new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stop_async())
    loop.close()
    
    return jsonify({"message": "Monitoring stopped", "status": "success"})

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to Website Proctor'})
    
    # Send current data if available
    global proctor_instance
    if proctor_instance:
        emit('initial_data', proctor_instance.get_dashboard_data())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_monitoring')
def handle_start_monitoring(data):
    """Handle start monitoring request from client"""
    url = data.get('url', 'https://example.com')
    global proctor_instance, is_monitoring
    
    if is_monitoring:
        emit('error', {'message': 'Already monitoring'})
        return
    
    proctor_instance = WebsiteProctor()
    
    # Start monitoring in background thread
    def run_monitor():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(proctor_instance.continuous_monitor(url))
        loop.close()
    
    monitor_thread = threading.Thread(target=run_monitor, daemon=True)
    monitor_thread.start()
    
    emit('monitoring_started', {'url': url, 'message': f'Started monitoring {url}'})

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Handle stop monitoring request from client"""
    global proctor_instance, is_monitoring
    
    if not is_monitoring:
        emit('error', {'message': 'Not monitoring'})
        return
    
    # Stop monitoring
    async def stop_async():
        if proctor_instance:
            await proctor_instance.stop_monitoring()
    
    # Run in new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stop_async())
    loop.close()
    
    emit('monitoring_stopped', {'message': 'Monitoring stopped'})

@socketio.on('get_verdict')
def handle_get_verdict():
    """Handle verdict request"""
    global proctor_instance
    
    if proctor_instance:
        verdict = proctor_instance.generate_verdict()
        dashboard_data = proctor_instance.get_dashboard_data()
        
        emit('verdict_generated', {
            'verdict': verdict,
            'summary': {
                'total_issues': len(dashboard_data['issues']),
                'major_issues': len([i for i in dashboard_data['issues'] if i['severity'] == 'major']),
                'minor_issues': len([i for i in dashboard_data['issues'] if i['severity'] == 'minor']),
                'suggestions_count': len(dashboard_data['suggestions'])
            }
        })
    else:
        emit('error', {'message': 'No monitoring data available'})

@socketio.on('test_azure')
def handle_test_azure():
    """Handle Azure connection test request from client (SocketIO)"""
    global azure_config
    
    # Test all Azure connections
    test_results = azure_config.test_all_connections()
    # Emit results back to the client
    emit('azure_test_results', test_results)

def open_browser_to_dashboard():
    """Open browser to dashboard after short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

def main():
    """Main entry point"""
    print("🔍 Website Proctor Starting...")
    print("📊 Dashboard will open at: http://localhost:5000")
    print("⚡ Real-time monitoring with Azure Computer Vision")
    print("🔧 Instructions:")
    print("   1. Open the dashboard in your browser")
    print("   2. Enter a website URL to monitor")
    print("   3. Click 'Start Monitoring'")
    print("   4. The system will open the website (80% screen) with console (20%)")
    print("   5. Dashboard shows real-time analysis")
    print("   6. Click 'Generate Verdict' when done")
    print("\n" + "="*60)
    
    # Open browser to dashboard in background
    browser_thread = threading.Thread(target=open_browser_to_dashboard, daemon=True)
    browser_thread.start()
    
    # Configure static files
    app.static_folder = '.'
    app.static_url_path = ''
    
    # Start Flask-SocketIO server
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Website Proctor...")
        # Cleanup
        global proctor_instance
        if proctor_instance:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(proctor_instance.stop_monitoring())
            loop.close()

if __name__ == "__main__":
    main()