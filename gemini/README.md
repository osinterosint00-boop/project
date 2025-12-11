# PyImgScan GUI

A user-friendly desktop application for scanning and analyzing documents from images.

## Features

- **Document Scanning:** Automatically detects the corners of a document in an image and corrects the perspective to produce a flat, top-down view.
- **Glare Removal:** A tool to automatically detect and remove bright glare spots from the image.
- **Compression Analysis:** An integrated tool to analyze the effects of JPEG compression. It allows you to:
    - Target specific file sizes (30KB, 100KB, 500KB, 1MB) or run a full analysis.
    - View the visually compressed image in the main window (for single-target analysis).
    - Access a detailed report with objective quality metrics (PSNR, SSIM, MSE).
    - Visualize the quality-size trade-off with a rate-distortion plot.
    - Get subjective analysis and recommendations for optimal compression.
- **Undo/Redo History:** Easily revert or re-apply changes using dedicated "Undo" and "Redo" buttons.
- **Change Picture:** Load a new image to work on without restarting the application.
- **Save Your Work:** Save the final edited image to your computer.

## Requirements

- Python 3.x
- The Python packages listed in `requirements-gui.txt`.

## How to Run

1.  **Install Dependencies:**
    Before running the application, make sure you have all the necessary packages installed. If you are in a virtual environment, make sure it's activated.

    ```bash
    pip install -r ../requirements-gui.txt
    ```

2.  **Run the Application:**

    ```bash
    python3 gui.py
    ```

## How to Use

1.  **Select an Image:**
    - On the welcome screen, click "Select Image" to choose a document image from your computer.

2.  **Edit the Image:**
    - The image will appear in the main editor window.
    - Use the tools in the left sidebar to process the image:
        - **Detect & Crop:** Straightens the document.
        - **Analyze Compression:** Opens a pop-up where you can select a target file size for analysis (e.g., 30KB, 100KB) or click "Run All & Plot" for a comprehensive analysis.
            - For single-target analysis, the compressed image will be displayed in the main window.
            - After any analysis, the "Show Analysis Report" button will become active to view the detailed report and plot.
        - **Remove Glare:** Reduces bright spots on the document.
        - **Change Picture:** Load a new image into the editor.
    - Use the buttons at the bottom to manage your workflow:
        - **Undo:** Reverts the last action.
        - **Redo:** Re-applies the last undone action.
        - **Save Image:** Saves the currently displayed image to a file.