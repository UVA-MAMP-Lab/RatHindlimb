#!/usr/bin/env python
"""Watch model_edits.qmd for changes and automatically re-render."""

import time
import sys
import os
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import threading

class Colors:
    """Terminal colors for pretty output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

class QuartoWatcher(FileSystemEventHandler):
    """Watches for changes to model_edits.qmd and triggers re-render."""
    
    def __init__(self, file_to_watch='model_edits.qmd'):
        self.file_to_watch = file_to_watch
        self.last_modified = 0
        self.debounce_seconds = 1.0  # Wait 1 second after last change
        self.render_in_progress = False
        self.pending_render = False
        
        print(f"{Colors.CYAN}{Colors.BOLD}╔════════════════════════════════════════════╗{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}║   Quarto Document Watcher (with Cache)     ║{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}╚════════════════════════════════════════════╝{Colors.END}")
        print(f"\n{Colors.YELLOW}Watching:{Colors.END} {file_to_watch}")
        print(f"{Colors.YELLOW}Cache:   {Colors.END} _freeze/execute/")
        print(f"{Colors.YELLOW}Output:  {Colors.END} model_edits_executed.html")
        print(f"\n{Colors.GREEN}✓ Watcher started!{Colors.END}")
        print(f"{Colors.CYAN}Press Ctrl+C to stop...{Colors.END}\n")
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        # Check if it's the file we're watching
        src_path = str(event.src_path)
        file_path = Path(src_path)
        if file_path.name != self.file_to_watch:
            return
        
        # Debounce: ignore if modified too recently
        current_time = time.time()
        if current_time - self.last_modified < self.debounce_seconds:
            return
        
        self.last_modified = current_time
        
        # If a render is in progress, mark that we need another render
        if self.render_in_progress:
            self.pending_render = True
            print(f"{Colors.YELLOW}⟳ Change detected, will render after current render completes...{Colors.END}")
            return
        
        # Trigger render in a separate thread
        threading.Thread(target=self.render_document).start()
    
    def render_document(self):
        """Render the Quarto document."""
        self.render_in_progress = True
        
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n{Colors.BLUE}═══════════════════════════════════════════{Colors.END}")
            print(f"{Colors.BOLD}[{timestamp}] Change detected! Re-rendering...{Colors.END}")
            print(f"{Colors.BLUE}═══════════════════════════════════════════{Colors.END}\n")
            
            # Step 1: Convert QMD to notebook
            print(f"{Colors.YELLOW}1/3 Converting QMD to notebook...{Colors.END}")
            result = subprocess.run(
                ['quarto', 'convert', self.file_to_watch],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                print(f"{Colors.RED}✗ Conversion failed:{Colors.END}")
                print(result.stderr)
                return
            print(f"{Colors.GREEN}  ✓ Converted{Colors.END}")
            
            # Step 2: Execute notebook (with caching)
            print(f"{Colors.YELLOW}2/3 Executing notebook (with cache)...{Colors.END}")
            result = subprocess.run(
                ['.venv/bin/python', 'execute_notebook.py'],
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=1800  # 30 minutes max for expensive cells
            )
            if result.returncode != 0:
                print(f"{Colors.RED}✗ Execution failed!{Colors.END}")
                return
            
            # Step 3: Render to HTML
            print(f"\n{Colors.YELLOW}3/3 Rendering to HTML...{Colors.END}")
            result = subprocess.run(
                ['quarto', 'render', 'model_edits_executed.ipynb', '--to', 'html'],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                print(f"{Colors.RED}✗ Render failed:{Colors.END}")
                print(result.stderr)
                return
            
            # Success!
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Render complete at {timestamp}!{Colors.END}")
            print(f"{Colors.CYAN}Refresh your browser to see changes.{Colors.END}\n")
            
        except subprocess.TimeoutExpired:
            print(f"{Colors.RED}✗ Render timed out!{Colors.END}\n")
        except Exception as e:
            print(f"{Colors.RED}✗ Error: {e}{Colors.END}\n")
        finally:
            self.render_in_progress = False
            
            # Check if another render is needed
            if self.pending_render:
                self.pending_render = False
                print(f"{Colors.YELLOW}Starting queued render...{Colors.END}")
                threading.Thread(target=self.render_document).start()

def main():
    """Main entry point."""
    # Make sure we're in the right directory
    script_path = Path(__file__)
    script_dir = script_path.parent.absolute()
    os.chdir(str(script_dir))
    
    # Check if the file exists
    if not Path('model_edits.qmd').exists():
        print(f"{Colors.RED}Error: model_edits.qmd not found in current directory!{Colors.END}")
        sys.exit(1)
    
    # Initial render
    print(f"{Colors.YELLOW}Performing initial render...{Colors.END}\n")
    watcher = QuartoWatcher()
    watcher.render_document()
    
    # Set up the observer
    observer = Observer()
    observer.schedule(watcher, path=str(Path('.')), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Stopping watcher...{Colors.END}")
        observer.stop()
        observer.join()
        print(f"{Colors.GREEN}✓ Watcher stopped. Goodbye!{Colors.END}\n")

if __name__ == "__main__":
    main()
