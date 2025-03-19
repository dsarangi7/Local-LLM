import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
import threading
from queue import Queue
import traceback
import sys
import os
from main import DocumentProcessor, OLLAMA_MODEL, check_ollama_model

class SingletonGUI:
    _instance = None
    
    @classmethod
    def ensure_single_instance(cls):
        try:
            lock_file = "gui.lock"
            if os.path.exists(lock_file):
                with open(lock_file, 'r') as f:
                    pid = int(f.read().strip())
                try:
                    os.kill(pid, 0)
                    messagebox.showerror(
                        "Already Running",
                        "AYK Chatbot is already running. Please close the existing window first."
                    )
                    sys.exit(1)
                except OSError:
                    os.remove(lock_file)
            
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
                
            import atexit
            def cleanup():
                try:
                    os.remove(lock_file)
                except:
                    pass
            atexit.register(cleanup)
            
        except Exception as e:
            print(f"Error in singleton check: {e}")
            sys.exit(1)

class ModernUI:
    PRIMARY = "#6B4EFF"
    SECONDARY = "#4A90E2"
    WHITE = "#FFFFFF"
    LIGHT_GRAY = "#F5F5F7"
    DARK_GRAY = "#666666"
    BG_COLOR = "#1E1E1E"
    TEXT_COLOR = "#FFFFFF"
    ACCENT_COLOR = "#00FF9D"
    
    TITLE_FONT = ("Helvetica", 24, "bold")
    HEADER_FONT = ("Helvetica", 16, "bold")
    NORMAL_FONT = ("Helvetica", 12)
    
    @staticmethod
    def create_rounded_frame(parent, **kwargs):
        frame = tk.Frame(parent, **kwargs)
        return frame

class RAGGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AYK AI Assistant")
        self.root.geometry("1000x800")
        self.root.configure(bg=ModernUI.BG_COLOR)
        
        self.processor = None
        self.initialization_thread = None
        self.is_initializing = False
        
        self.main_frame = tk.Frame(root, bg=ModernUI.BG_COLOR, padx=20, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        header_frame = tk.Frame(self.main_frame, bg=ModernUI.BG_COLOR)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="AYK AI Assistant",
            font=ModernUI.TITLE_FONT,
            fg=ModernUI.ACCENT_COLOR,
            bg=ModernUI.BG_COLOR
        )
        title_label.pack(side=tk.LEFT)
        
        self.status_label = tk.Label(
            header_frame,
            text=f"● Offline",
            font=ModernUI.NORMAL_FONT,
            fg=ModernUI.DARK_GRAY,
            bg=ModernUI.BG_COLOR
        )
        self.status_label.pack(side=tk.RIGHT, pady=10)
        
        self.init_button = tk.Button(
            self.main_frame,
            text="Initialize AYK",
            command=self._safe_initialize_processor,
            font=ModernUI.NORMAL_FONT,
            bg=ModernUI.PRIMARY,
            fg=ModernUI.WHITE,
            activebackground=ModernUI.SECONDARY,
            activeforeground=ModernUI.WHITE,
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        self.init_button.pack(pady=(0, 20))
        
        chat_frame = tk.Frame(self.main_frame, bg=ModernUI.BG_COLOR)
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=ModernUI.NORMAL_FONT,
            bg=ModernUI.LIGHT_GRAY,
            fg=ModernUI.DARK_GRAY,
            height=20,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        self.chat_display.tag_configure("user", foreground="#0084FF", font=("Helvetica", 12, "bold"))
        self.chat_display.tag_configure("assistant", foreground="#00B386", font=("Helvetica", 12))
        self.chat_display.tag_configure("system", foreground="#666666", font=("Helvetica", 10, "italic"))
        
        input_frame = tk.Frame(self.main_frame, bg=ModernUI.BG_COLOR)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.query_entry = tk.Entry(
            input_frame,
            font=ModernUI.NORMAL_FONT,
            bg=ModernUI.LIGHT_GRAY,
            fg=ModernUI.DARK_GRAY,
            relief=tk.FLAT,
            insertbackground=ModernUI.DARK_GRAY
        )
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.query_button = tk.Button(
            input_frame,
            text="Send",
            command=self._safe_query,
            font=ModernUI.NORMAL_FONT,
            bg=ModernUI.PRIMARY,
            fg=ModernUI.WHITE,
            activebackground=ModernUI.SECONDARY,
            activeforeground=ModernUI.WHITE,
            relief=tk.FLAT,
            padx=20
        )
        self.query_button.pack(side=tk.RIGHT)
        
        self.query_entry.config(state='disabled')
        self.query_button.config(state='disabled')
        
        self.query_entry.bind('<Return>', lambda e: self._safe_query())
        
        self.chat_display.insert(tk.END, "Welcome to AYK AI Assistant!\n", "system")
        self.chat_display.insert(tk.END, "Please initialize the system to begin.\n\n", "system")
        self.chat_display.config(state='disabled')
        
    def _add_message(self, message, tag):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{message}\n\n", tag)
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')
        
    def _safe_initialize_processor(self):
        if self.is_initializing:
            return
            
        self.is_initializing = True
        self.init_button.config(state='disabled')
        self.status_label.config(text="● Initializing...", fg="#FFA500")
        self._add_message("Initializing AYK AI system...", "system")
        
        def init_thread():
            try:
                if not check_ollama_model():
                    raise RuntimeError("Ollama server is not running or not responding. Please start it with 'ollama serve'.")
                
                self._add_message("Ollama server detected, proceeding...", "system")
                self.processor = DocumentProcessor(
                    data_dir="C:/Users/dsara/Desktop/Local LLM/data",
                    db_dir="db"
                )
                self._add_message("Loading knowledge base...", "system")
                
                documents = self.processor.load_and_process_documents()
                self._add_message("Processing documents...", "system")
                
                self.processor.initialize_vector_store(documents)
                self._add_message("Organizing information...", "system")
                
                self.processor.initialize_rag()
                
                self.root.after(0, self._on_initialization_complete)
                
            except Exception as e:
                self.root.after(0, self._on_initialization_error, str(e))
            finally:
                self.is_initializing = False
        
        self.initialization_thread = threading.Thread(target=init_thread)
        self.initialization_thread.daemon = True
        self.initialization_thread.start()
    
    def _on_initialization_complete(self):
        self.status_label.config(text="● Online", fg="#00FF00")
        self.query_entry.config(state='normal')
        self.query_button.config(state='normal')
        self.init_button.config(state='normal')
        self._add_message("AYK AI is ready! How can I help you today?", "assistant")
    
    def _on_initialization_error(self, error_msg):
        self.status_label.config(text="● Offline", fg="#FF0000")
        self.init_button.config(state='normal')
        self._add_message(f"Initialization Error: {error_msg}", "system")
        messagebox.showerror("Initialization Error", str(error_msg))
    
    def _safe_query(self):
        if not self.processor:
            messagebox.showerror("Error", "AYK AI is not initialized")
            return
            
        query = self.query_entry.get().strip()
        if not query:
            return
            
        self.query_entry.delete(0, tk.END)
        self.query_entry.config(state='disabled')
        self.query_button.config(state='disabled')
        
        self._add_message(f"You: {query}", "user")
        self._add_message("AYK is thinking...", "system")
        
        def query_thread():
            try:
                response, _ = self.processor.query_rag(query)
                self.root.after(0, self._on_query_complete, response)
            except Exception as e:
                self.root.after(0, self._on_query_error, str(e))
        
        threading.Thread(target=query_thread, daemon=True).start()
    
    def _on_query_complete(self, response):
        self.chat_display.config(state='normal')
        self.chat_display.delete("end-3c linestart", "end-1c lineend+1c")
        self.chat_display.config(state='disabled')
        
        self._add_message(f"AYK: {response}", "assistant")
        
        self.query_entry.config(state='normal')
        self.query_button.config(state='normal')
        self.query_entry.focus()
    
    def _on_query_error(self, error_msg):
        self._add_message(f"Error: {error_msg}", "system")
        self.query_entry.config(state='normal')
        self.query_button.config(state='normal')
        messagebox.showerror("Query Error", str(error_msg))

def main():
    SingletonGUI.ensure_single_instance()
    root = tk.Tk()
    app = RAGGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()