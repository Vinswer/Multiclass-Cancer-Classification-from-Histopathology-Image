import os
import io
import zipfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from difflib import get_close_matches

class DataExplorer:
    def __init__(self, train_zip, train_csv, test_zip=None, output_dir="outputs", show_plots=True):
        self.train_zip = zipfile.ZipFile(train_zip, 'r')
        self.test_zip = zipfile.ZipFile(test_zip, 'r') if test_zip else None
        self.labels_df = pd.read_csv(train_csv)

        self.output_dir = output_dir
        self.show_plots = show_plots
        os.makedirs(self.output_dir, exist_ok=True)

        # ---- Pre-build zip index (supports subdirectories, case-insensitive) ----
        self._train_members = [n for n in self.train_zip.namelist() if not n.endswith("/")]
        # Create mappings for both original names and basenames (without directories), all lowercase for matching
        self._member_set_lower = set([m.lower() for m in self._train_members])
        self._basename_to_member = {}
        for m in self._train_members:
            self._basename_to_member.setdefault(os.path.basename(m).lower(), m)

        # Common file extensions
        self._exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")

    def _savefig(self, filename, dpi=200, bbox_inches="tight"):
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        # if self.show_plots:
        #     plt.show()
        # else:
        #     plt.close()
        print(f"[Saved] {save_path}")
        return save_path

    def _get_id_series(self):
        """Automatically identify the column in CSV that represents files"""
        for col in ["id", "image_id", "file_name", "filename", "Image", "ImageID"]:
            if col in self.labels_df.columns:
                return self.labels_df[col]
        # Raise error if not found (more user-friendly than KeyError)
        raise ValueError(
            f"No image column found in CSV, available columns: {list(self.labels_df.columns)}. "
            "Please ensure it contains one of: id/image_id/file_name/filename."
        )

    def _resolve_zip_member(self, image_id):
        """
        Resolve the actual member name in zip based on CSV's image_id:
        - Direct match
        - Try appending common extensions
        - Basename match (ignore subdirectories)
        - Fuzzy match (return closest match)
        """
        if not isinstance(image_id, str):
            image_id = str(image_id)

        cand = image_id
        cand_lower = cand.lower()

        # If CSV already contains extension or subdirectory, try direct match first
        if cand_lower in self._member_set_lower:
            # Find the original member with correct case
            for m in self._train_members:
                if m.lower() == cand_lower:
                    return m

        # If no extension, try appending extensions (considering subdirectory paths)
        root, ext = os.path.splitext(cand_lower)
        if ext == "":
            for e in self._exts:
                name_e = root + e
                # 1) Direct full path match
                if name_e in self._member_set_lower:
                    for m in self._train_members:
                        if m.lower() == name_e:
                            return m
                # 2) Basename only match
                if name_e in self._basename_to_member:
                    return self._basename_to_member[name_e]

        # If CSV contains basename (with extension), try basename -> member
        base = os.path.basename(cand_lower)
        if base in self._basename_to_member:
            return self._basename_to_member[base]

        # Fuzzy match (last resort)
        close = get_close_matches(base, [os.path.basename(m).lower() for m in self._train_members], n=1, cutoff=0.9)
        if close:
            return self._basename_to_member[close[0]]

        return None

    def _open_image_from_zip(self, zip_file, image_id):
        member = self._resolve_zip_member(image_id)
        if member is None:
            raise FileNotFoundError(f"Image '{image_id}' not found in zip.")
        with zip_file.open(member) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def explore_dataset_statistics(self):
        print("DATASET EXPLORATION")
        if self.labels_df is not None:
            print(f"\nDataset Statistics:")
            total = len(self.labels_df)
            print(f"Total samples: {total}")

            if "label" in self.labels_df.columns:
                label_counts = self.labels_df["label"].value_counts().sort_index()
                benign_cnt = int(label_counts.get(0, 0))
                malignant_cnt = int(label_counts.get(1, 0))

                print(f"\nClass Distribution:")
                if total > 0:
                    print(f"Benign (0): {benign_cnt:,} ({benign_cnt / total * 100:.1f}%)")
                    print(f"Malignant (1): {malignant_cnt:,} ({malignant_cnt / total * 100:.1f}%)")
                else:
                    print("Empty dataset.")

                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                label_counts.plot(kind='bar', color=['skyblue', 'lightcoral'][:len(label_counts)])
                plt.title('Class Distribution')
                plt.xlabel('Class (0=Benign, 1=Malignant)')
                plt.ylabel('Count')
                plt.xticks(rotation=0)

                plt.subplot(1, 2, 2)
                plt.pie(
                    [benign_cnt, malignant_cnt],
                    labels=['Benign', 'Malignant'],
                    autopct='%1.1f%%',
                    colors=['skyblue', 'lightcoral']
                )
                plt.title('Class Proportion')

                plt.tight_layout()
                self._savefig("class_distribution.png")
            else:
                print("CSV does not contain 'label' column, skipping class distribution plot.")

    def visualize_sample_images(self, n=10, seed=42, filename_prefix="samples"):
        print(f"\nVisualizing {n} random samples:")
        ids = self._get_id_series()
        samples = self.labels_df.sample(min(n, len(self.labels_df)), random_state=seed)

        cols = min(5, len(samples))
        rows = (len(samples) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        sample_count = 0
        missing = 0
        for _, row in samples.iterrows():
            img_id = None
            for col in ["id", "image_id", "file_name", "filename", "Image", "ImageID"]:
                if col in row:
                    img_id = row[col]
                    break
            if img_id is None:
                continue

            try:
                img = self._open_image_from_zip(self.train_zip, img_id)
            except FileNotFoundError:
                missing += 1
                continue

            ax = axes[sample_count]
            ax.imshow(img)
            # Optional: display label
            if "label" in row:
                label_text = "Malignant" if int(row["label"]) == 1 else "Benign"
                title_color = "red" if int(row["label"]) == 1 else "green"
                ax.set_title(label_text, color=title_color, fontsize=10)
            ax.axis("off")
            sample_count += 1
            if sample_count == len(axes):
                break

        # Fill remaining empty subplots
        for j in range(sample_count, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        out = self._savefig(f"{filename_prefix}_{sample_count}.png")
        if missing:
            print(f"[Warning] {missing} sample(s) were missing in the zip and were skipped.")
        return out

    def analyze_image_properties(self, n=200, filename_prefix="image_properties"):
        print("Analyzing image sizes:")
        if len(self.labels_df) == 0:
            print("Empty dataframe.")
            return None
        ids = self._get_id_series()
        samples = self.labels_df.sample(min(n, len(self.labels_df)))
        sizes = []
        missing = 0
        for _, row in samples.iterrows():
            img_id = None
            for col in ["id", "image_id", "file_name", "filename", "Image", "ImageID"]:
                if col in row:
                    img_id = row[col]
                    break
            if img_id is None:
                continue
            try:
                img = self._open_image_from_zip(self.train_zip, img_id)
                sizes.append(img.size)  # (w, h)
            except FileNotFoundError:
                missing += 1
                continue

        if len(sizes) == 0:
            print("No images could be opened from the sample.")
            if missing:
                print(f"[Hint] {missing} image id(s) not found in zip. "
                      f"Please verify if CSV columns are filenames or need extension appending, or if images are in subdirectories.")
            return None

        widths = [w for w, h in sizes]
        heights = [h for w, h in sizes]
        print(f"Sampled {len(sizes)} images (missing skipped: {missing})")
        print(f"Width  mean={np.mean(widths):.1f}, min={min(widths)}, max={max(widths)}")
        print(f"Height mean={np.mean(heights):.1f}, min={min(heights)}, max={max(heights)}")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(widths, bins=20)
        plt.title("Width Distribution")
        plt.xlabel("Width (px)")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        plt.hist(heights, bins=20)
        plt.title("Height Distribution")
        plt.xlabel("Height (px)")
        plt.ylabel("Count")

        plt.tight_layout()
        return self._savefig(f"{filename_prefix}.png")


