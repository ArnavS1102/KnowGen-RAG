import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from PIL import Image
import pymupdf
import torch
import torch.nn as nn
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList, VisionEncoderDecoderModel

from dotenv import load_dotenv

load_dotenv(".env")
class RunningVarTorch:
    def __init__(self, L=15, norm=False):
        self.values = None
        self.L = L
        self.norm = norm

    def push(self, x: torch.Tensor):
        assert x.dim() == 1
        if self.values is None:
            self.values = x[:, None]
        elif self.values.shape[1] < self.L:
            self.values = torch.cat((self.values, x[:, None]), 1)
        else:
            self.values = torch.cat((self.values[:, 1:], x[:, None]), 1)

    def variance(self):
        if self.values is None:
            return
        if self.norm:
            return torch.var(self.values, 1) / self.values.shape[1]
        else:
            return torch.var(self.values, 1)

class StoppingCriteriaScores(StoppingCriteria):
    def __init__(self, threshold: float = 0.015, window_size: int = 200):
        super().__init__()
        self.threshold = threshold
        self.vars = RunningVarTorch(norm=True)
        self.varvars = RunningVarTorch(L=window_size)
        self.stop_inds = defaultdict(int)
        self.stopped = defaultdict(bool)
        self.size = 0
        self.window_size = window_size

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_scores = scores[-1]
        self.vars.push(last_scores.max(1)[0].float().cpu())
        self.varvars.push(self.vars.variance())
        self.size += 1
        if self.size < self.window_size:
            return False

        varvar = self.varvars.variance()
        for b in range(len(last_scores)):
            if varvar[b] < self.threshold:
                if self.stop_inds[b] > 0 and not self.stopped[b]:
                    self.stopped[b] = self.stop_inds[b] >= self.size
                else:
                    self.stop_inds[b] = int(
                        min(max(self.size, 1) * 1.15 + 150 + self.window_size, 4095)
                    )
            else:
                self.stop_inds[b] = 0
                self.stopped[b] = False
        return all(self.stopped.values()) and len(self.stopped) > 0

class PDF2MD:
    def __init__(self):
        self.nougat_filepath = os.getenv("NOUGAT_PATH")
        self.processor = AutoProcessor.from_pretrained(self.nougat_filepath)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.nougat_filepath)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def rasterize_paper(self, pdf: Path, outpath: Optional[Path] = None, dpi: int = 96, return_pil=False, pages=None) -> Optional[List[io.BytesIO]]:
        pillow_images = []
        if outpath is None:
            return_pil = True
        try:
            if isinstance(pdf, (str, Path)):
                pdf = pymupdf.open(pdf)
            if pages is None:
                pages = range(len(pdf))
            for i in pages:
                page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
                if return_pil:
                    pillow_images.append(io.BytesIO(page_bytes))
                else:
                    with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                        f.write(page_bytes)
        except Exception:
            pass
        if return_pil:
            return pillow_images

    def parse_dir(self, dest_dir, source_dir):
        files_skipped = []
        cleaned_md_base = dest_dir

        for index, filename in enumerate(os.listdir(source_dir)):
            try:
                print(f'File Number: {index}')
                print(filename)
                md_filename = filename.replace("pdf", "md")

                md_exists = md_filename in os.listdir(dest_dir)

                if md_exists:
                    print(f"Skipping {filename} as Markdown file already exists.")
                    files_skipped.append(filename)
                    continue

                new_md_file_path = f'{dest_dir}/{md_filename}'

                with open(new_md_file_path, 'w') as md_file:
                    filepath = f'{source_dir}/{filename}'
                    images = self.rasterize_paper(pdf=filepath, return_pil=True)

                    for i, image in enumerate(images):
                        try:
                            image = Image.open(image)
                            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
                            outputs = self.model.generate(
                                pixel_values.to(self.device),
                                min_length=1,
                                max_length=3584,
                                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                return_dict_in_generate=True,
                                output_scores=True,
                                stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]))

                            generated = self.processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
                            generated = self.processor.post_process_generation(generated, fix_markdown=False)

                            md_file.write(generated + '\n')

                        except Exception as e:
                            print(f"Error processing image {i} in file {filename}: {e}")
                            files_skipped.append(filename)
                            break

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                files_skipped.append(filename)
                continue
