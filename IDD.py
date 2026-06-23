import os
import fitz
import cv2
import numpy as np
from PIL import Image
import io
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import models, transforms
import argparse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import tempfile
from reportlab.lib.units import inch
import imagehash
from scipy.spatial import distance

class ImprovedDocumentImageDeduplicator:
    def __init__(self):
        # Initialize pre-trained model (optional, can be disabled)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model.to(self.device)
            self.use_deep_learning = True
        except:
            self.use_deep_learning = False
            print("Deep learning model not available, using perceptual hashing only")
        
        # Image preprocessing for deep learning
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Store extracted image information
        self.images_info = []
        
    def extract_images_from_pdf(self, file_path):
        """Extract images and their positions from PDF"""
        doc = fitz.open(file_path)
        images_info = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)
            
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Get image position information
                image_info = {
                    "source": "pdf",
                    "page": page_num,
                    "index": img_index,
                    "data": image_bytes,
                    "format": image_ext,
                    "position": {"page": page_num, "index": img_index}
                }
                images_info.append(image_info)
        
        doc.close()
        return images_info
    
    def extract_images_from_docx(self, file_path):
        """Extract images from Word document"""
        doc = Document(file_path)
        images_info = []
        rels = doc.part.rels
        
        for rel in rels:
            if "image" in rels[rel].target_ref:
                image_part = rels[rel].target_part
                image_bytes = image_part.blob
                
                image_info = {
                    "source": "docx",
                    "page": 0,
                    "index": len(images_info),
                    "data": image_bytes,
                    "format": self._get_image_format(image_bytes),
                    "position": {"index": len(images_info)}
                }
                images_info.append(image_info)
        
        return images_info
    
    def _get_image_format(self, image_bytes):
        """Determine image format from byte data"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return image.format.lower()
        except:
            return "unknown"
    
    def process_document(self, file_path):
        """Process document and extract all images"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            self.images_info = self.extract_images_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            self.images_info = self.extract_images_from_docx(file_path)
        else:
            raise ValueError("Unsupported document format: " + ext)
        
        print(f"Extracted {len(self.images_info)} images from document")
        
        # Calculate perceptual hashes for all images
        for img_info in self.images_info:
            img_info["phash"] = self.calculate_perceptual_hash(img_info["data"])
            img_info["ahash"] = self.calculate_average_hash(img_info["data"])
        
        return self.images_info
    
    def calculate_perceptual_hash(self, image_bytes, hash_size=16):
        """Calculate perceptual hash (pHash) for image"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            phash = imagehash.phash(image, hash_size=hash_size)
            return str(phash)
        except Exception as e:
            print(f"Perceptual hash calculation error: {e}")
            return None
    
    def calculate_average_hash(self, image_bytes, hash_size=8):
        """Calculate average hash (aHash) for image"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            ahash = imagehash.average_hash(image, hash_size=hash_size)
            return str(ahash)
        except Exception as e:
            print(f"Average hash calculation error: {e}")
            return None
    
    def hamming_distance(self, hash1, hash2):
        """Calculate Hamming distance between two hashes"""
        if hash1 is None or hash2 is None:
            return float('inf')
        return sum(ch1 != ch2 for ch1, ch2 in zip(hash1, hash2))
    
    def calculate_structural_similarity(self, img1_bytes, img2_bytes):
        """Calculate Structural Similarity Index (SSIM) between two images"""
        try:
            img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return 0
            
            # Resize to same dimensions
            h1, w1 = img1.shape
            h2, w2 = img2.shape
            h, w = min(h1, h2), min(w1, w2)
            
            img1_resized = cv2.resize(img1, (w, h))
            img2_resized = cv2.resize(img2, (w, h))
            
            # Calculate SSIM
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            mu1 = cv2.GaussianBlur(img1_resized, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(img2_resized, (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(img1_resized ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(img2_resized ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(img1_resized * img2_resized, (11, 11), 1.5) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return np.mean(ssim_map)
        except Exception as e:
            print(f"SSIM calculation error: {e}")
            return 0
    
    def find_duplicate_regions_improved(self, img1_bytes, img2_bytes, min_match_area_ratio=0.3):
        """Find duplicate regions with improved accuracy using multiple methods"""
        img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            return None, 0, 0, 0  # 修复：返回4个值
        
        # Store original dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Method 1: Template matching for large duplicate regions
        scale = 0.5  # Scale down for faster processing
        template = cv2.resize(img1, (int(w1 * scale), int(h1 * scale)))
        search_img = cv2.resize(img2, (int(w2 * scale), int(h2 * scale)))
        
        # Convert to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        
        # Template matching
        result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Check if we have a good template match
        template_match_quality = max_val
        
        # Method 2: Improved ORB with spatial consistency check
        orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)
        
        # Detect keypoints and compute descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, 0, template_match_quality, 0  # 修复：返回4个值
        
        # Use BFMatcher with ratio test
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test (Lowe's ratio test)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 10:
            return None, len(good_matches), template_match_quality, 0  # 修复：返回4个值
        
        # Check spatial consistency - matches should be in similar relative positions
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            # No consistent transformation found
            return None, len(good_matches), template_match_quality, 0  # 修复：返回4个值
        
        # Count inliers
        inlier_count = np.sum(mask)
        
        # Calculate area of matched region (approximate)
        min_area_ratio = 0
        composite = None
        
        if inlier_count >= 10:
            # Create convex hull of matched points
            try:
                src_inliers = src_pts[mask.ravel() == 1]
                dst_inliers = dst_pts[mask.ravel() == 1]
                
                if len(src_inliers) > 4:
                    # Calculate convex hull area ratio
                    hull_src = cv2.convexHull(src_inliers)
                    hull_dst = cv2.convexHull(dst_inliers)
                    
                    area_src = cv2.contourArea(hull_src)
                    area_dst = cv2.contourArea(hull_dst)
                    
                    # Calculate area ratios
                    area_ratio_src = area_src / (w1 * h1)
                    area_ratio_dst = area_dst / (w2 * h2)
                    
                    min_area_ratio = min(area_ratio_src, area_ratio_dst)
                    
                    # Create composite visualization
                    composite = self.create_composite_visualization(
                        img1, img2, kp1, kp2, good_matches, mask
                    )
            except Exception as e:
                print(f"Error in area calculation: {e}")
                min_area_ratio = 0
        print("composite, inlier_count, template_match_quality, min_area_ratio:",composite, inlier_count, template_match_quality, min_area_ratio)
        return composite, inlier_count, template_match_quality, min_area_ratio
    
    def create_composite_visualization(self, img1, img2, kp1, kp2, matches, mask):
        """Create visualization of matches"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        composite_height = max(h1, h2)
        composite_width = w1 + w2
        composite_img = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        composite_img[:h1, :w1] = img1
        composite_img[:h2, w1:w1+w2] = img2
        
        # Draw only inlier matches
        for i, match in enumerate(matches):
            if i < len(mask) and mask[i] == 1:
                # Get the matching keypoints for each of the images
                img1_idx = match.queryIdx
                img2_idx = match.trainIdx
                
                # Get the coordinates
                pt1 = (int(kp1[img1_idx].pt[0]), int(kp1[img1_idx].pt[1]))
                pt2 = (int(kp2[img2_idx].pt[0]) + w1, int(kp2[img2_idx].pt[1]))
                
                # Draw the line on the composite image
                cv2.line(composite_img, pt1, pt2, (0, 255, 0), 1)
                cv2.circle(composite_img, pt1, 3, (0, 0, 255), -1)
                cv2.circle(composite_img, pt2, 3, (0, 0, 255), -1)
        
        # Convert BGR to RGB
        composite_img_rgb = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
        return composite_img_rgb
    
    def compare_all_images_improved(self, 
                                  phash_threshold=5,
                                  ahash_threshold=10,
                                  min_matches=20,
                                  min_area_ratio=0.2,
                                  min_ssim=0.8,
                                  deep_similarity_threshold=0.9):
        """Improved image comparison using multiple methods"""
        results = []
        n = len(self.images_info)
        
        print(f"Comparing {n} images using multiple methods...")
        
        # Extract deep learning features if enabled
        if self.use_deep_learning:
            deep_features = []
            valid_indices = []
            
            for i, img_info in enumerate(self.images_info):
                feat = self.extract_deep_features(img_info["data"])
                if feat is not None:
                    deep_features.append(feat)
                    valid_indices.append(i)
            
            if deep_features:
                deep_features = np.array(deep_features)
                deep_similarity = cosine_similarity(deep_features)
            else:
                deep_similarity = None
        else:
            deep_similarity = None
        
        # Compare all image pairs
        for i in range(n):
            for j in range(i + 1, n):
                img1_info = self.images_info[i]
                img2_info = self.images_info[j]
                
                # 1. Check perceptual hash (pHash) - very sensitive
                phash_dist = self.hamming_distance(img1_info.get("phash"), img2_info.get("phash"))
                if phash_dist <= phash_threshold:
                    # Very similar images, likely duplicates
                    composite, matches, template_quality, area_ratio = self.find_duplicate_regions_improved(
                        img1_info["data"], img2_info["data"], min_area_ratio
                    )
                    
                    result = {
                        "image1_index": i,
                        "image2_index": j,
                        "similarity_method": "pHash",
                        "phash_distance": phash_dist,
                        "ahash_distance": self.hamming_distance(img1_info.get("ahash"), img2_info.get("ahash")),
                        "num_matches": matches,
                        "template_match_quality": template_quality,
                        "area_ratio": area_ratio,
                        "composite_image": composite,
                        "image1_info": img1_info,
                        "image2_info": img2_info
                    }
                    results.append(result)
                    print(f"Found duplicate (pHash): Images {i} and {j}, pHash dist: {phash_dist}")
                    continue
                
                # 2. Check average hash (aHash) - less sensitive but faster
                ahash_dist = self.hamming_distance(img1_info.get("ahash"), img2_info.get("ahash"))
                if ahash_dist <= ahash_threshold:
                    # Check with more detailed methods
                    composite, matches, template_quality, area_ratio = self.find_duplicate_regions_improved(
                        img1_info["data"], img2_info["data"], min_area_ratio
                    )
                    
                    if matches >= min_matches and area_ratio >= min_area_ratio:
                        result = {
                            "image1_index": i,
                            "image2_index": j,
                            "similarity_method": "aHash+ORB",
                            "phash_distance": phash_dist,
                            "ahash_distance": ahash_dist,
                            "num_matches": matches,
                            "template_match_quality": template_quality,
                            "area_ratio": area_ratio,
                            "composite_image": composite,
                            "image1_info": img1_info,
                            "image2_info": img2_info
                        }
                        results.append(result)
                        print(f"Found duplicate (aHash+ORB): Images {i} and {j}, aHash dist: {ahash_dist}, matches: {matches}, area ratio: {area_ratio:.3f}")
                    continue
                
                # 3. Check deep learning similarity if available
                if deep_similarity is not None and i in valid_indices and j in valid_indices:
                    idx_i = valid_indices.index(i)
                    idx_j = valid_indices.index(j)
                    deep_sim = deep_similarity[idx_i, idx_j]
                    
                    if deep_sim > deep_similarity_threshold:
                        # Check with detailed methods
                        composite, matches, template_quality, area_ratio = self.find_duplicate_regions_improved(
                            img1_info["data"], img2_info["data"], min_area_ratio
                        )
                        
                        if matches >= min_matches and area_ratio >= min_area_ratio:
                            result = {
                                "image1_index": i,
                                "image2_index": j,
                                "similarity_method": "DeepLearning+ORB",
                                "deep_similarity": deep_sim,
                                "phash_distance": phash_dist,
                                "ahash_distance": ahash_dist,
                                "num_matches": matches,
                                "template_match_quality": template_quality,
                                "area_ratio": area_ratio,
                                "composite_image": composite,
                                "image1_info": img1_info,
                                "image2_info": img2_info
                            }
                            results.append(result)
                            print(f"Found duplicate (Deep+ORB): Images {i} and {j}, deep sim: {deep_sim:.3f}, matches: {matches}, area ratio: {area_ratio:.3f}")
                        continue
                
                # 4. Direct ORB matching for remaining pairs (most computationally expensive)
                # Only do this for images that might be similar based on size
                try:
                    img1 = Image.open(io.BytesIO(img1_info["data"]))
                    img2 = Image.open(io.BytesIO(img2_info["data"]))
                    
                    # Skip if aspect ratios are very different
                    aspect1 = img1.width / img1.height if img1.height > 0 else 0
                    aspect2 = img2.width / img2.height if img2.height > 0 else 0
                    
                    if max(aspect1, aspect2) > 0 and abs(aspect1 - aspect2) / max(aspect1, aspect2) < 0.3:  # Within 30% aspect ratio difference
                        composite, matches, template_quality, area_ratio = self.find_duplicate_regions_improved(
                            img1_info["data"], img2_info["data"], min_area_ratio
                        )
                        
                        if (matches >= min_matches * 1.5 and area_ratio >= min_area_ratio * 1.5) or \
                           (template_quality > 0.8 and area_ratio >= min_area_ratio):
                            # Calculate SSIM as additional check
                            ssim = self.calculate_structural_similarity(img1_info["data"], img2_info["data"])
                            
                            if ssim >= min_ssim or (matches >= min_matches * 2 and area_ratio >= min_area_ratio):
                                result = {
                                    "image1_index": i,
                                    "image2_index": j,
                                    "similarity_method": "ORB+Template",
                                    "phash_distance": phash_dist,
                                    "ahash_distance": ahash_dist,
                                    "num_matches": matches,
                                    "template_match_quality": template_quality,
                                    "area_ratio": area_ratio,
                                    "ssim": ssim,
                                    "composite_image": composite,
                                    "image1_info": img1_info,
                                    "image2_info": img2_info
                                }
                                results.append(result)
                                print(f"Found duplicate (ORB): Images {i} and {j}, matches: {matches}, area ratio: {area_ratio:.3f}, SSIM: {ssim:.3f}")
                except Exception as e:
                    print(f"Error processing images {i} and {j}: {e}")
                    continue
        
        return results
    
    def extract_deep_features(self, image_bytes):
        """Extract image features using pre-trained model (optional)"""
        if not self.use_deep_learning:
            return None
            
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
            
            return features.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Deep feature extraction error: {e}")
            return None
    
    def generate_report(self, results, output_path):
        """Generate PDF report"""
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        temp_files = []

        # Title
        title = Paragraph("AI Image Duplication Report (Improved Algorithm)", styles["Title"])
        story.append(title)
        story.append(Spacer(1, 12))

        if not results:
            story.append(Paragraph("No duplicate images found.", styles["BodyText"]))
            doc.build(story)
            return

        # Summary
        summary = Paragraph(f"Found {len(results)} pairs of duplicate images:", styles["Heading2"])
        story.append(summary)
        story.append(Spacer(1, 12))

        # Page settings
        page_width, page_height = letter
        left_margin = right_margin = 1 * inch
        available_width = page_width - left_margin - right_margin
        max_image_height = 5 * inch

        try:
            for i, result in enumerate(results):
                # Duplicate pair information
                info_text = (f"Duplicate pair #{i+1}: Image {result['image1_index']} "
                           f"(Page {result['image1_info']['page']+1}) "
                           f"and Image {result['image2_index']} "
                           f"(Page {result['image2_info']['page']+1})<br/>"
                           f"Detection Method: {result.get('similarity_method', 'Unknown')}<br/>"
                           f"pHASH Distance: {result.get('phash_distance', 'N/A')}<br/>"
                           f"Matches: {result.get('num_matches', 'N/A')}<br/>"
                           f"Matched Area Ratio: {result.get('area_ratio', 0):.3f}")
                
                if 'ssim' in result:
                    info_text += f"<br/>SSIM: {result['ssim']:.3f}"
                if 'deep_similarity' in result:
                    info_text += f"<br/>Deep Similarity: {result['deep_similarity']:.3f}"
                
                story.append(Paragraph(info_text, styles["BodyText"]))
                story.append(Spacer(1, 12))
            
                # Add composite image if available
                if result.get('composite_image') is not None:
                    comp_img = Image.fromarray(result['composite_image'])
                    comp_width, comp_height = comp_img.size
                    comp_aspect_ratio = comp_height / comp_width
                    
                    comp_display_width = min(available_width, comp_width / 72 * inch)
                    comp_display_height = comp_display_width * comp_aspect_ratio
                    
                    if comp_display_height > max_image_height:
                        scale_factor = max_image_height / comp_display_height
                        comp_display_width *= scale_factor
                        comp_display_height = max_image_height
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                        comp_img.save(tmp.name, 'PNG')
                        temp_files.append(tmp.name)
                        comp_img_element = RLImage(tmp.name, width=comp_display_width, height=comp_display_height)
                        story.append(comp_img_element)
                        
                        caption = Paragraph("Composite image showing matched regions (green lines connect matching points)", 
                                          styles["Italic"])
                        story.append(caption)
                
                if i < len(results) - 1:
                    story.append(PageBreak())

            doc.build(story)
    
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(description="AI Image Duplication Detection Tool")
    parser.add_argument("input_file", help="Input Word or PDF document path")
    parser.add_argument("--output", "-o", default="improved_report.pdf", 
                       help="Output report path")
    parser.add_argument("--phash_threshold", type=int, default=5,
                       help="Perceptual hash Hamming distance threshold (lower = stricter)")
    parser.add_argument("--ahash_threshold", type=int, default=10,
                       help="Average hash Hamming distance threshold")
    parser.add_argument("--min_matches", type=int, default=20,
                       help="Minimum number of ORB feature matches")
    parser.add_argument("--min_area", type=float, default=0.2,
                       help="Minimum area ratio of matched region (0-1)")
    parser.add_argument("--min_ssim", type=float, default=0.8,
                       help="Minimum Structural Similarity Index (0-1)")
    
    args = parser.parse_args()
    
    # Initialize deduplicator
    deduplicator = ImprovedDocumentImageDeduplicator()
    
    try:
        # Process document
        print("Extracting images from document...")
        images_info = deduplicator.process_document(args.input_file)
        
        if not images_info:
            print("No images found in document.")
            return
        
        # Compare images
        print("Comparing images with improved algorithm...")
        results = deduplicator.compare_all_images_improved(
            phash_threshold=args.phash_threshold,
            ahash_threshold=args.ahash_threshold,
            min_matches=args.min_matches,
            min_area_ratio=args.min_area,
            min_ssim=args.min_ssim
        )
        
        # Generate report
        print("Generating report...")
        deduplicator.generate_report(results, args.output)
        
        print(f"Complete! Report saved to: {args.output}")
        print(f"Found {len(results)} duplicate pairs")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
