"""Extract charts from PDF files using LayoutParser and save them as images."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Union

import layoutparser as lp
from pdf2image import convert_from_path
import torch

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class ChartExtractor(BaseReader):
    """Extract charts from PDF files using LayoutParser and save them as images."""

    def __init__(self, output_dir: str = "extracted_charts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = lp.Detectron2LayoutModel(
            config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def load_data(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Extract charts from the PDF file and save them as images in the output directory."""
        return self.load(file_path, metadata=metadata, extra_info=extra_info)

    def load(
        self,
        file_path: Union[Path, str],
        metadata: bool = True,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Extract charts from a single PDF file."""
        # Convert PDF pages to images
        pages = convert_from_path(file_path)
        documents = []

        for page_num, page in enumerate(pages):
            # Detect the layout of the page
            layout = self.model.detect(page)
            # Filter for 'Figure' type blocks
            figure_blocks = [block for block in layout if block.type == "Figure"]
            for idx, block in enumerate(figure_blocks):
                # Get block coordinates and ensure they are within image bounds
                x1, y1, x2, y2 = map(int, block.coordinates)
                width, height = page.size
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                # Crop the figure region from the page
                figure_image = page.crop((x1, y1, x2, y2))
                # Save the figure image
                figure_filename = f'figure_{os.path.basename(file_path)}_page{page_num + 1}_{idx + 1}.png'
                figure_path = os.path.join(self.output_dir, figure_filename)
                figure_image.save(figure_path)
                # Optionally, create a Document for each image (empty text)
                if metadata:
                    doc_extra_info = extra_info.copy() if extra_info else {}
                    doc_extra_info.update(
                        {
                            "file_path": str(file_path),
                            "page_number": page_num + 1,
                            "figure_index": idx + 1,
                            "figure_path": figure_path,
                        }
                    )
                    documents.append(
                        Document(
                            text="",  # No text, since it's an image
                            extra_info=doc_extra_info,
                        )
                    )
        return documents
