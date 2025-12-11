import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image
import cv2
import numpy as np
import io
import threading
from tkinter import messagebox
from cvtools import resize, perspective_transform, getoutlines, simple_erode, simple_dilate, brightness_contrast, blank
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PyImgScan GUI")
        self.geometry("1024x768")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.welcome_frame = WelcomeFrame(self, self.show_editor)
        self.welcome_frame.pack(fill="both", expand=True)

        self.editor_frame = None

        self.mainloop()

    def show_editor(self, filepath):
        self.welcome_frame.pack_forget()
        self.editor_frame = EditorFrame(self, filepath)
        self.editor_frame.pack(fill="both", expand=True)

class WelcomeFrame(ctk.CTkFrame):
    def __init__(self, master, on_image_select):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self.on_image_select = on_image_select

        self.welcome_label = ctk.CTkLabel(self, text="Welcome to PyImgScan", font=ctk.CTkFont(size=30, weight="bold"))
        self.welcome_label.pack(pady=50)

        self.select_image_button = ctk.CTkButton(self, text="Select Image", command=self.select_image)
        self.select_image_button.pack(pady=20)

    def select_image(self):
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if filepath:
            self.on_image_select(filepath)

class EditorFrame(ctk.CTkFrame):
    def __init__(self, master, filepath):
        super().__init__(master)

        self.image_history = []
        self.redo_history = []
        self.analysis_results = None
        self.analysis_toplevel = None

        # Grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)

        self.sidebar_title = ctk.CTkLabel(self.sidebar_frame, text="Tools", font=ctk.CTkFont(size=20, weight="bold"))
        self.sidebar_title.pack(pady=20)

        self.detect_corners_button = ctk.CTkButton(self.sidebar_frame, text="Detect & Crop", command=self.detect_and_crop)
        self.detect_corners_button.pack(pady=10, padx=20, fill="x")

        self.analyze_button = ctk.CTkButton(self.sidebar_frame, text="Analyze Compression", command=self.open_analysis_options)
        self.analyze_button.pack(pady=10, padx=20, fill="x")

        self.glare_button = ctk.CTkButton(self.sidebar_frame, text="Remove Glare", command=self.remove_glare)
        self.glare_button.pack(pady=10, padx=20, fill="x")

        self.show_report_button = ctk.CTkButton(self.sidebar_frame, text="Show Analysis Report", state="disabled", command=self.show_analysis_report)
        self.show_report_button.pack(pady=10, padx=20, fill="x")
        
        self.change_picture_button = ctk.CTkButton(self.sidebar_frame, text="Change Picture", command=self.change_picture)
        self.change_picture_button.pack(side="bottom", pady=20, padx=20, fill="x")


        # Image display
        self.image_label = ctk.CTkLabel(self, text="")
        self.image_label.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # Bottom bar
        self.bottom_bar_frame = ctk.CTkFrame(self, height=80, corner_radius=0)
        self.bottom_bar_frame.grid(row=1, column=1, sticky="nsew", padx=20, pady=20)
        self.bottom_bar_frame.grid_propagate(False)
        
        self.bottom_bar_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.back_button = ctk.CTkButton(self.bottom_bar_frame, text="Undo", command=self.undo)
        self.back_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.redo_button = ctk.CTkButton(self.bottom_bar_frame, text="Redo", command=self.redo, state="disabled")
        self.redo_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.save_button = ctk.CTkButton(self.bottom_bar_frame, text="Save Image", command=self.save_image)
        self.save_button.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        # Load initial image and set state
        self.current_image = Image.open(filepath)
        self.add_to_history(self.current_image)
        self.display_image(self.current_image)
        self.update_button_states()

    def display_image(self, pil_image):
        self.update_idletasks()
        w, h = self.image_label.winfo_width(), self.image_label.winfo_height()
        
        if w < 2 or h < 2:
            w, h = 800, 600

        resized_image = pil_image.copy()
        resized_image.thumbnail((w - 40, h - 40), Image.LANCZOS)
        
        ctk_image = ctk.CTkImage(light_image=resized_image, dark_image=resized_image, size=resized_image.size)
        self.image_label.configure(image=ctk_image)
        self.current_image = pil_image

    def add_to_history(self, image):
        self.image_history.append(image.copy())
        self.redo_history.clear()
        self.update_button_states()


    def detect_and_crop(self):
        self.add_to_history(self.current_image)
        
        open_cv_image = np.array(self.current_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        img_adj, scale, _, img_edge = self.preprocess(open_cv_image)
        img_hull = self.gethull(img_edge)
        corners = self.getcorners(img_hull)

        if corners is None:
            messagebox.showerror("Error", "Could not detect the four corners of the document. Please try a different image.")
            return

        corners = corners.reshape(4, 2) * scale
        img_corrected = perspective_transform(img_adj, corners)
        img_corrected_pil = Image.fromarray(cv2.cvtColor(img_corrected, cv2.COLOR_BGR2RGB))
        
        self.display_image(img_corrected_pil)

    def update_and_display_compressed(self, compressed_image):
        self.add_to_history(self.current_image)
        self.display_image(compressed_image)

    def open_analysis_options(self):
        if self.analysis_toplevel is None or not self.analysis_toplevel.winfo_exists():
            self.analysis_toplevel = AnalysisOptionsWindow(self)
        else:
            self.analysis_toplevel.focus()

    def show_analysis_report(self):
        if self.analysis_results:
            AnalysisReportWindow(self, self.analysis_results)

    def run_analysis_thread(self, targets_kb):
        thread = threading.Thread(target=self.run_analysis, args=(targets_kb,))
        thread.daemon = True
        thread.start()

    def run_analysis(self, targets_kb):
        try:
            self.analysis_results = []
            original_image_cv = np.array(self.image_history[0])
            original_image_cv_rgb = original_image_cv[:, :, ::-1]

            for target_kb in targets_kb:
                target_bytes = target_kb * 1024
                best_quality = -1
                best_size_diff = float('inf')
                
                low = 1
                high = 100
                found_quality = -1

                while low <= high:
                    quality = (low + high) // 2
                    if quality == 0: break
                    
                    buffer = io.BytesIO()
                    self.image_history[0].save(buffer, "JPEG", quality=quality)
                    size_in_bytes = buffer.tell()
                    
                    size_diff = abs(size_in_bytes - target_bytes)

                    if size_diff < best_size_diff:
                        best_size_diff = size_diff
                        found_quality = quality

                    if size_in_bytes > target_bytes:
                        high = quality - 1
                    elif size_in_bytes < target_bytes:
                        low = quality + 1
                    else:
                        break 

                buffer = io.BytesIO()
                self.image_history[0].save(buffer, "JPEG", quality=found_quality)
                compressed_image = Image.open(buffer)
                compressed_image_cv = np.array(compressed_image)

                psnr_val = psnr(original_image_cv_rgb, compressed_image_cv)
                
                h, w, _ = original_image_cv_rgb.shape
                min_dim = min(h, w)
                win_size = min(7, min_dim)
                if win_size % 2 == 0:
                    win_size -= 1
                
                if win_size < 3:
                    ssim_val = 0.0
                else:
                    ssim_val = ssim(original_image_cv_rgb, compressed_image_cv, win_size=win_size, channel_axis=-1, data_range=255)

                mse_val = mse(original_image_cv_rgb, compressed_image_cv)

                self.analysis_results.append({
                    "target_kb": target_kb,
                    "actual_kb": buffer.tell() / 1024,
                    "quality": found_quality,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "mse": mse_val
                })

                if len(targets_kb) == 1:
                    self.after(0, self.update_and_display_compressed, compressed_image)

            self.show_report_button.configure(state="normal")
            messagebox.showinfo("Analysis Complete", "The compression analysis is complete. Click 'Show Analysis Report' to see the results.")
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{e}")


    def remove_glare(self):
        self.add_to_history(self.current_image)
        
        open_cv_image = np.array(self.current_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)
        inpainted_image = cv2.inpaint(open_cv_image, mask, 3, cv2.INPAINT_NS)
        inpainted_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
        
        self.display_image(inpainted_pil)

    def undo(self):
        if len(self.image_history) > 1:
            self.redo_history.append(self.image_history.pop())
            self.display_image(self.image_history[-1])
        self.update_button_states()

    def redo(self):
        if self.redo_history:
            self.image_history.append(self.redo_history.pop())
            self.display_image(self.image_history[-1])
        self.update_button_states()
        
    def change_picture(self):
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if filepath:
            self.image_history.clear()
            self.redo_history.clear()
            self.analysis_results = None
            
            self.current_image = Image.open(filepath)
            self.add_to_history(self.current_image)
            self.display_image(self.current_image)
            
            self.show_report_button.configure(state="disabled")
            self.update_button_states()

    def update_button_states(self):
        self.back_button.configure(state="normal" if len(self.image_history) > 1 else "disabled")
        self.redo_button.configure(state="normal" if self.redo_history else "disabled")

    def save_image(self):
        filepath = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=(("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All files", "*.*"))
        )
        if filepath:
            self.current_image.save(filepath)

    def preprocess(self, img):
        img_adj = brightness_contrast(img, 1.56, -60)
        scale = img_adj.shape[0] / 500.0
        img_scaled = resize(img_adj, height=500)
        img_gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (11, 11), 0)
        img_edge = cv2.Canny(img_gray, 60, 245)
        img_edge = simple_dilate(img_edge)
        return img_adj, scale, img_scaled, img_edge

    def gethull(self, img_edge):
        img_prehull = img_edge.copy()
        outlines = getoutlines(img_prehull)
        img_hull = blank(img_prehull.shape, img_prehull.dtype, "0")
        for outline in range(len(outlines)):
            hull = cv2.convexHull(outlines[outline])
            cv2.drawContours(img_hull, [hull], 0, 255, 3)
        img_hull = simple_erode(img_hull)
        return img_hull

    def getcorners(self, img_hull):
        img_outlines = img_hull.copy()
        outlines = getoutlines(img_outlines)
        outlines = sorted(outlines, key=cv2.contourArea, reverse=True)[:4]
        corners = None
        for outline in outlines:
            perimeter = cv2.arcLength(outline, True)
            approx = cv2.approxPolyDP(outline, 0.02 * perimeter, True)
            if len(approx) == 4:
                corners = approx
                break
        return corners

class AnalysisOptionsWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.title("Analysis Options")
        self.geometry("300x350")
        self.transient(master)
        
        self.label = ctk.CTkLabel(self, text="Select a target file size:")
        self.label.pack(pady=10)

        self.buttons = []
        targets = {"30 KB": 30, "100 KB": 100, "500 KB": 500, "1 MB": 1024}
        for text, size_kb in targets.items():
            btn = ctk.CTkButton(self, text=text, command=lambda s=size_kb: self.start_analysis(s))
            btn.pack(pady=5, padx=20, fill="x")
            self.buttons.append(btn)
            
        self.run_all_button = ctk.CTkButton(self, text="Run All & Plot", command=self.run_all_analysis)
        self.run_all_button.pack(pady=15, padx=20, fill="x")
        self.buttons.append(self.run_all_button)

        self.progress_bar = ctk.CTkProgressBar(self, mode='indeterminate')
        
        self.after(20, self.grab_set)

    def start_analysis(self, target_size_kb):
        for btn in self.buttons:
            btn.configure(state="disabled")
        self.progress_bar.pack(pady=10, padx=20, fill="x")
        self.progress_bar.start()
        
        self.master.run_analysis_thread([target_size_kb])
        self.destroy()

    def run_all_analysis(self):
        for btn in self.buttons:
            btn.configure(state="disabled")
        self.progress_bar.pack(pady=10, padx=20, fill="x")
        self.progress_bar.start()

        targets_kb = [30, 100, 500, 1024]
        self.master.run_analysis_thread(targets_kb)
        self.destroy()

class AnalysisReportWindow(ctk.CTkToplevel):
    def __init__(self, master, results):
        super().__init__(master)
        self.title("Analysis Report")
        self.geometry("850x700")
        self.transient(master)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.plot_label = ctk.CTkLabel(self, text="")
        self.plot_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.generate_and_display_plot(results)

        self.textbox = ctk.CTkTextbox(self, wrap="word")
        self.textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.generate_report_text(results)
        self.textbox.configure(state="disabled")

        self.after(20, self.grab_set)

    def generate_and_display_plot(self, results):
        sizes = [r['actual_kb'] for r in results]
        psnr_vals = [r['psnr'] for r in results]

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.plot(sizes, psnr_vals, 'w-o', label='PSNR')
        ax.set_xlabel("File Size (KB)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Rate-Distortion Curve")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        
        if len(sizes) > 2:
            slopes = [(psnr_vals[i+1] - psnr_vals[i]) / (sizes[i+1] - sizes[i]) for i in range(len(sizes)-1)]
            if slopes:
                optimal_index = np.argmax(np.diff(slopes)) + 1 if len(slopes) > 1 else 0
                ax.plot(sizes[optimal_index], psnr_vals[optimal_index], 'r*', markersize=15, label='Optimal Point')
                ax.legend()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_image = Image.open(buf)
        
        ctk_plot = ctk.CTkImage(light_image=plot_image, dark_image=plot_image, size=plot_image.size)
        self.plot_label.configure(image=ctk_plot)
        plt.close(fig)

    def generate_report_text(self, results):
        report = "Compression & Rate-Distortion Analysis\n"
        report += "=" * 40 + "\n\n"

        report += f"{ 'Target Size':<15}{'Actual Size':<15}{'Quality':<10}{'PSNR (dB)':<15}{'SSIM':<10}{'MSE':<10}\n"
        report += "-" * 75 + "\n"

        for r in sorted(results, key=lambda x: x['actual_kb']):
            report += f"{r['target_kb']:<15.0f}{r['actual_kb']:<15.2f}{r['quality']:<10}{r['psnr']:<15.2f}{r['ssim']:<10.4f}{r['mse']:<10.2f}\n"

        report += "\n\n" + "=" * 40 + "\n"
        report += "Subjective Analysis\n"
        report += "=" * 40 + "\n\n"

        for r in sorted(results, key=lambda x: x['actual_kb']):
            report += f"--- {r['target_kb']} KB Target ---\n"
            if r['actual_kb'] < 50:
                report += "Visual Quality: Poor.\nArtifacts: Heavy blocking, ringing, and color banding are very noticeable.\n\n"
            elif r['actual_kb'] < 200:
                report += "Visual Quality: Acceptable.\nArtifacts: Minor blocking and softness are visible upon inspection.\n\n"
            elif r['actual_kb'] < 800:
                report += "Visual Quality: Good.\nArtifacts: Very few artifacts. Slight softness might be visible in high-frequency areas.\n\n"
            else:
                report += "Visual Quality: Excellent.\nArtifacts: Essentially none.\n\n"
        
        report += "\n" + "=" * 40 + "\n"
        report += "Recommendation\n"
        report += "=" * 40 + "\n\n"
        report += "The 'Optimal Point' on the plot marks the 'knee' of the curve, where increasing the file size yields diminishing returns in quality.\n"
        report += "For a good balance of size and quality, the setting closest to this point is recommended. For high-quality archival, use the highest setting."

        self.textbox.insert("0.0", report)

if __name__ == "__main__":
    App()