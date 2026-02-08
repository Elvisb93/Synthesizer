import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import queue
import threading
import os
from typing import List, Dict, Any
from core.models import GeneratorConfig, ColumnDefinition, ColumnType, ColumnConstraints, AIProvider
from core.controller import GeneratorController

class ColumnWidget(ttk.Frame):
    def __init__(self, master, index: int, on_remove: callable, column_def: ColumnDefinition = None):
        super().__init__(master, padding="5")
        self.index = index
        self.on_remove = on_remove
        
        # Row layout
        # self.prompt_var is no longer used for the Text widget binding directly
            self.initial_prompt = column_def.prompt_instruction if column_def else ""
            
            self.regex_var = tk.StringVar(value=column_def.constraints.regex_pattern or "") if column_def else tk.StringVar()
            self.logic_var = tk.StringVar(value=column_def.constraints.expression or "") if column_def else tk.StringVar()
            self.sim_var = tk.StringVar(value=str(column_def.constraints.similarity_threshold) if column_def and column_def.constraints.similarity_threshold is not None else "") if column_def else tk.StringVar()
            # self.allow_dupes_var will be init below with default False
        
        # Grid layout
        # Row 0: Basic Info
        ttk.Label(self, text="Name:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.name_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(self, text="Type:").grid(row=0, column=2, sticky="w")
        type_cb = ttk.Combobox(self, textvariable=self.type_var, values=[e.value for e in ColumnType], state="readonly", width=15)
        type_cb.grid(row=0, column=3, padx=5)
        
        ttk.Label(self, text="Prompt:").grid(row=0, column=4, sticky="nw", pady=3)
        self.columnconfigure(5, weight=1)
        
        # Replace Entry with Text widget for dynamic resizing
        self.prompt_text = tk.Text(self, height=1, width=60, undo=True)
        self.prompt_text.grid(row=0, column=5, padx=5, sticky="ew", columnspan=2, pady=2)
        self.prompt_text.insert("1.0", self.initial_prompt)
        self.prompt_text.bind("<KeyRelease>", self._adjust_height)
        
        # Trigger initial height adjustment
        self._adjust_height()

        ttk.Button(self, text="Remove", command=lambda: self.on_remove(self)).grid(row=0, column=7, padx=5)

        # Row 1: Advanced Constraints (Expandable? For now just row 2)
        advanced_frame = ttk.Frame(self)
        advanced_frame.grid(row=1, column=0, columnspan=8, sticky="w", pady=2)
        
        ttk.Label(advanced_frame, text="Regex:").pack(side="left", padx=(0, 2))
        ttk.Entry(advanced_frame, textvariable=self.regex_var, width=10).pack(side="left", padx=(0, 10))

        ttk.Label(advanced_frame, text="Logic (e.g. after @[Date]):").pack(side="left", padx=(0, 2))
        ttk.Entry(advanced_frame, textvariable=self.logic_var, width=15).pack(side="left", padx=(0, 10))
        
        ttk.Label(advanced_frame, text="Sim. Thresh (0-1):").pack(side="left", padx=(0, 2))
        ttk.Entry(advanced_frame, textvariable=self.sim_var, width=5).pack(side="left", padx=(0, 10))

        self.allow_dupes_var = tk.BooleanVar(value=column_def.constraints.allow_duplicates if column_def and column_def.constraints else False)
        ttk.Checkbutton(advanced_frame, text="Allow Duplicates", variable=self.allow_dupes_var).pack(side="left")

    def _adjust_height(self, event=None):
        """Dynamically adjust the height of the prompt text widget based on content."""
        # Get count of lines
        content = self.prompt_text.get("1.0", "end-1c")
        
        # Calculate number of lines needed based on newlines and wrapping
        # This is a bit rough, but 'count' from tk.Text isn't always reliable for wrapped lines 
        # unless update_idletasks is called, which can be slow.
        # Simple approach: Newlines + rough char count
        
        lines = content.count('\n') + 1
        
        # Also check for wrapping (very rough approx: chars / width)
        # width = self.prompt_text.cget("width")
        # if width > 0:
        #     wrapped_lines = len(content) // width
        #     lines += wrapped_lines
            
        # Let's trust the widget's internal wrapping if we can, or just stick to newline counting for now
        # to avoid jitter. If user wants more space, they usually hit enter.
        
        # Clamp between 1 and 5
        new_height = max(1, min(lines, 5))
        
        if int(self.prompt_text.cget("height")) != new_height:
            self.prompt_text.config(height=new_height)

        
    def get_definition(self) -> ColumnDefinition:
        sim_val = self.sim_var.get()
        sim_float = float(sim_val) if sim_val.strip() else None
        
        return ColumnDefinition(
            name=self.name_var.get(),
            type=ColumnType(self.type_var.get()),
            prompt_instruction=self.prompt_text.get("1.0", "end-1c").strip(),
            constraints=ColumnConstraints(
                regex_pattern=self.regex_var.get() if self.regex_var.get().strip() else None,
                expression=self.logic_var.get() if self.logic_var.get().strip() else None,
                similarity_threshold=sim_float,
                allow_duplicates=self.allow_dupes_var.get()
            ) 
        )

class MainWindow(tk.Tk):
    def __init__(self, controller: GeneratorController):
        super().__init__()
        self.controller = controller
        self.title("Synthetic Data Generator (LM Studio)")
        self.geometry("1400x900")
        
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.model_queue = queue.Queue() # New queue for models
        
        self.column_widgets: List[ColumnWidget] = []
        self.imported_data: List[Dict[str, Any]] = None
        
        self._setup_ui()
        self._start_queue_processing()
        self._init_controller_callbacks()
        
    def _setup_ui(self):
        # Top: Configuration
        config_frame = ttk.LabelFrame(self, text="Configuration", padding="10")
        config_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(config_frame, text="Model ID:").pack(side="left")
        self.model_id_var = tk.StringVar(value="local-model")
        # Changed to Combobox
        self.model_combo = ttk.Combobox(config_frame, textvariable=self.model_id_var, width=30)
        self.model_combo.pack(side="left", padx=5)
        
        ttk.Button(config_frame, text="Refresh Models", command=self._refresh_models).pack(side="left", padx=5)
        
        ttk.Button(config_frame, text="Save Config", command=self.save_config).pack(side="left", padx=5)
        ttk.Button(config_frame, text="Load Config", command=self.load_config).pack(side="left", padx=5)
        ttk.Button(config_frame, text="Import Data", command=self.import_data).pack(side="left", padx=5)
        ttk.Button(config_frame, text="Docs / Help", command=self._open_help).pack(side="left", padx=5)
        
        ttk.Label(config_frame, text="Rows:").pack(side="left", padx=(20, 0))
        self.rows_var = tk.IntVar(value=10)
        ttk.Entry(config_frame, textvariable=self.rows_var, width=8).pack(side="left", padx=5)
        
        ttk.Label(config_frame, text="Sim. Threshold:").pack(side="left", padx=5)
        self.sim_var = tk.DoubleVar(value=0.85)
        ttk.Entry(config_frame, textvariable=self.sim_var, width=6).pack(side="left", padx=5)

        ttk.Label(config_frame, text="Max Retries:").pack(side="left", padx=5)
        self.retries_var = tk.IntVar(value=50)
        ttk.Entry(config_frame, textvariable=self.retries_var, width=6).pack(side="left", padx=5)

        # AI Configuration Section
        self.ai_config_frame = ttk.LabelFrame(self, text="🤖 AI Configuration", padding="10")
        self.ai_config_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(self.ai_config_frame, text="Provider:").pack(side="left")
        self.provider_var = tk.StringVar(value=AIProvider.LM_STUDIO.value)
        provider_combo = ttk.Combobox(
            self.ai_config_frame, 
            textvariable=self.provider_var, 
            values=[p.value for p in AIProvider], 
            state="readonly", 
            width=20
        )
        provider_combo.pack(side="left", padx=5)
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)
        
        ttk.Label(self.ai_config_frame, text="API Key:").pack(side="left", padx=(20, 0))
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(self.ai_config_frame, textvariable=self.api_key_var, width=30, show="*")
        self.api_key_entry.pack(side="left", padx=5)
        
        self.test_connection_btn = ttk.Button(self.ai_config_frame, text="Test Connection", command=self._test_connection)
        self.test_connection_btn.pack(side="left", padx=5)
        
        # Azure-specific fields (shown only for Azure OpenAI)
        self.azure_frame = ttk.LabelFrame(self, text="Azure OpenAI Configuration", padding="10")
        # Don't pack yet - will be shown/hidden by _on_provider_change
        
        ttk.Label(self.azure_frame, text="Azure Endpoint:").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=5)
        self.azure_endpoint_var = tk.StringVar()
        azure_endpoint_entry = ttk.Entry(self.azure_frame, textvariable=self.azure_endpoint_var, width=40)
        azure_endpoint_entry.grid(row=0, column=1, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(self.azure_frame, text="(e.g., https://your-resource.openai.azure.com)").grid(row=0, column=4, sticky="w", padx=5)
        
        ttk.Label(self.azure_frame, text="Deployment Name:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=5)
        self.azure_deployment_var = tk.StringVar()
        azure_deployment_entry = ttk.Entry(self.azure_frame, textvariable=self.azure_deployment_var, width=25)
        azure_deployment_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        ttk.Label(self.azure_frame, text="(e.g., gpt-4, gpt-35-turbo)").grid(row=1, column=2, sticky="w", padx=5)
        
        # Initialize provider state (must be after azure_frame creation)
        self._on_provider_change(None)

        # Magic Generator
        magic_frame = ttk.LabelFrame(self, text="✨ Magic Generator", padding="10")
        magic_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(magic_frame, text="Describe your dataset:").pack(side="left")
        self.magic_prompt_var = tk.StringVar()
        ttk.Entry(magic_frame, textvariable=self.magic_prompt_var, width=60).pack(side="left", padx=5)
        
        self.magic_btn = ttk.Button(magic_frame, text="Auto-Generate Schema", command=self._on_magic_generate)
        self.magic_btn.pack(side="left", padx=5)


        # Middle: Columns
        columns_frame_container = ttk.LabelFrame(self, text="Columns", padding="10")
        columns_frame_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Scrollable area for columns
        self.canvas = tk.Canvas(columns_frame_container)
        scrollbar = ttk.Scrollbar(columns_frame_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add Column Button
        ttk.Button(columns_frame_container, text="+ Add Column", command=self._add_column).pack(side="bottom", pady=5)

        # Bottom: Actions & Log
        action_frame = ttk.Frame(self, padding="10")
        action_frame.pack(fill="x")
        
        self.start_btn = ttk.Button(action_frame, text="Start Generation", command=self.toggle_generation)
        self.start_btn.pack(side="left")
        
        # Export Menu
        self.export_var = tk.StringVar()
        self.export_btn = ttk.Menubutton(action_frame, text="Export Data...", state="disabled")
        export_menu = tk.Menu(self.export_btn, tearoff=0)
        export_menu.add_command(label="Export CSV", command=lambda: self.export_data("csv"))
        export_menu.add_command(label="Export JSON", command=lambda: self.export_data("json"))
        export_menu.add_command(label="Export SQL", command=lambda: self.export_data("sql"))
        export_menu.add_separator()
        export_menu.add_command(label="Export Quality Report (PDF)", command=lambda: self.export_data("pdf_report"))
        export_menu.add_command(label="Export Narrative (PDF)", command=lambda: self.export_data("pdf_narrative"))
        self.export_btn.config(menu=export_menu)
        self.export_btn.pack(side="left", padx=10)

        self.analyze_btn = ttk.Button(action_frame, text="Analyze Quality", command=self._on_analyze, state="disabled")
        self.analyze_btn.pack(side="left", padx=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(action_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=10)
        
        # Log
        log_frame = ttk.LabelFrame(self, text="Log", padding="5")
        log_frame.pack(fill="both", expand=False, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, state="disabled")
        self.log_text.pack(fill="both", expand=True)
        
        # Initial columns
        self._add_column()
        self._add_column()

    def _add_column(self, column_def: ColumnDefinition = None):
        col = ColumnWidget(self.scrollable_frame, len(self.column_widgets), self._remove_column, column_def)
        col.pack(fill="x", pady=2)
        self.column_widgets.append(col)
        
    def _remove_column(self, widget: ColumnWidget):
        widget.destroy()
        self.column_widgets.remove(widget)
        
    def _refresh_models(self):
        # Run in thread
        def task():
            try:
                # We need a temporary client to fetch models if controller isn't initialized
                # Or we can just use the controller's logic if we trust it
                from core.llm_client import LLMClient
                from core.models import GeneratorConfig
                
                # Create a temp client just for listing
                temp_config = GeneratorConfig(model_id="temp")
                client = LLMClient(temp_config)
                
                models = client.list_models()
                if models:
                    self.model_queue.put(models)
                    self.log_queue.put(f"Found {len(models)} models.")
                else:
                    self.log_queue.put("No models found or connection failed.")
            except Exception as e:
                self.log_queue.put(f"Error fetching models: {e}")
                 
        threading.Thread(target=task).start()

    def _on_magic_generate(self):
        prompt = self.magic_prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Input Required", "Please describe your dataset first.")
            return

        self.magic_btn.config(state="disabled", text="Generating...")
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"✨ Magic Generation started for: '{prompt}'...\n")
        self.log_text.config(state="disabled")

        def task():
            try:
                from core.llm_client import LLMClient
                from core.models import GeneratorConfig, ColumnDefinition, ColumnConstraints, ColumnType
                
                # Create temp client
                model_id = self.model_id_var.get()
                temp_config = GeneratorConfig(model_id=model_id)
                client = LLMClient(temp_config)
                
                # Call connection check first
                if not client.check_connection():
                    self.log_queue.put("Error: Could not connect to LM Studio.")
                    self.magic_btn.after(0, lambda: self.magic_btn.config(state="normal", text="Auto-Generate Schema"))
                    return

                schema_list = client.generate_schema(prompt)
                
                if not schema_list:
                    self.log_queue.put("Error: LLM returned empty or invalid schema.")
                else:
                    self.log_queue.put(f"✨ Success! Generated {len(schema_list)} columns.")
                    
                    # Process schema in UI thread
                    def update_ui():
                        # Clear existing
                        for widget in self.column_widgets[:]:
                            self._remove_column(widget)
                        
                        # Add new
                        for col_dict in schema_list:
                            try:
                                # Safe enum conversion
                                col_type_str = col_dict.get("type", "Short Text")
                                try:
                                    col_type = ColumnType(col_type_str)
                                except ValueError:
                                    # Fallback if LLM halluncinated a type
                                    col_type = ColumnType.SHORT_TEXT
                                    self.log_queue.put(f"Warning: Unknown type '{col_type_str}' for column '{col_dict.get('name')}', defaulting to Short Text.")

                                constraints_dict = col_dict.get("constraints", {})
                                constraints = ColumnConstraints(
                                    min_value=constraints_dict.get("min_value"),
                                    max_value=constraints_dict.get("max_value"),
                                    options=constraints_dict.get("options", []),
                                    min_length=constraints_dict.get("min_length", 10),
                                    max_length=constraints_dict.get("max_length", 2000),
                                    regex_pattern=constraints_dict.get("regex_pattern"),
                                    expression=constraints_dict.get("expression"),
                                    similarity_threshold=constraints_dict.get("similarity_threshold"),
                                    faker_provider=constraints_dict.get("faker_provider"),
                                    allow_duplicates=constraints_dict.get("allow_duplicates", False)
                                )

                                col_def = ColumnDefinition(
                                    name=col_dict.get("name", "untitled"),
                                    type=col_type,
                                    prompt_instruction=col_dict.get("prompt_instruction", ""),
                                    constraints=constraints
                                )
                                self._add_column(col_def)
                            except Exception as e:
                                self.log_queue.put(f"Error adding column: {e}")

                    self.magic_btn.after(0, update_ui)

            except Exception as e:
                self.log_queue.put(f"Magic Generation Error: {e}")
            finally:
                self.magic_btn.after(0, lambda: self.magic_btn.config(state="normal", text="Auto-Generate Schema"))

        threading.Thread(target=task).start()

    def _on_provider_change(self, event):
        """Show/hide API key and Azure fields based on provider selection."""
        provider = self.provider_var.get()
        if provider == AIProvider.LM_STUDIO.value:
            # Hide API key for local LM Studio
            self.api_key_entry.config(state="disabled")
            self.api_key_var.set("")
            # Hide Azure fields
            self.azure_frame.pack_forget()
        elif provider == AIProvider.AZURE_OPENAI.value:
            # Show API key for Azure
            self.api_key_entry.config(state="normal")
            # Show Azure configuration frame right after magic frame
            self.azure_frame.pack(fill="x", padx=10, pady=5, after=self.ai_config_frame)
        else:
            # Show API key for other cloud providers
            self.api_key_entry.config(state="normal")
            # Hide Azure fields
            self.azure_frame.pack_forget()
    
    def _test_connection(self):
        """Test connection with current provider settings."""
        try:
            provider_str = self.provider_var.get()
            provider = AIProvider(provider_str)
            api_key = self.api_key_var.get() if provider != AIProvider.LM_STUDIO else None
            model_id = self.model_id_var.get() or "test-model"
            
            # Create temp config
            temp_config = GeneratorConfig(
                model_id=model_id,
                provider=provider,
                api_key=api_key
            )
            
            # Test connection
            from core.llm_client import LLMClient
            client = LLMClient(temp_config)
            
            if client.check_connection():
                messagebox.showinfo("Success", f"✅ Connected to {provider_str} successfully!")
                self.log_queue.put(f"Connection test passed for {provider_str}")
            else:
                messagebox.showerror("Connection Failed", f"❌ Could not connect to {provider_str}. Check your settings.")
                self.log_queue.put(f"Connection test failed for {provider_str}")
                
        except ValueError as e:
            messagebox.showerror("Configuration Error", f"API key is required for cloud providers.\n\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"Connection test failed:\n{e}")
            self.log_queue.put(f"Connection error: {e}")

    def _init_controller_callbacks(self):
        self.controller.on_log = lambda msg: self.log_queue.put(msg)
        self.controller.on_progress = lambda curr, total: self.progress_queue.put((curr, total))
        self.controller.on_finished = lambda: self.progress_queue.put("DONE")

    def _start_queue_processing(self):
        # Process logs
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
            
        # Process progress
        while not self.progress_queue.empty():
            data = self.progress_queue.get_nowait()
            if data == "DONE":
                self.start_btn.config(text="Start Generation")
                self.export_btn.config(state="normal")
                self.analyze_btn.config(state="normal")
                messagebox.showinfo("Done", "Generation Complete!")
            else:
                curr, total = data
                self.progress_var.set((curr / total) * 100)
        
        # Process models
        while not self.model_queue.empty():
            models = self.model_queue.get_nowait()
            self.model_combo['values'] = models
            if models:
                self.model_combo.set(models[0])
                
        self.after(100, self._start_queue_processing)
        
    def toggle_generation(self):
        if self.controller.is_running:
            self.controller.stop_generation()
            self.start_btn.config(text="Start Generation")
            return

        # Gather Config
        try:
            config = GeneratorConfig(
                model_id=self.model_id_var.get(),
                provider=AIProvider(self.provider_var.get()),
                api_key=self.api_key_var.get() if self.provider_var.get() != AIProvider.LM_STUDIO.value else None,
                azure_endpoint=self.azure_endpoint_var.get() if self.provider_var.get() == AIProvider.AZURE_OPENAI.value else None,
                azure_deployment=self.azure_deployment_var.get() if self.provider_var.get() == AIProvider.AZURE_OPENAI.value else None,
                num_rows=self.rows_var.get(),
                similarity_threshold=self.sim_var.get(),
                max_retries=self.retries_var.get(),
                existing_data=getattr(self, 'imported_data', None)
            )
            columns = [w.get_definition() for w in self.column_widgets]
            
            if not columns:
                messagebox.showerror("Error", "Please add at least one column.")
                return

            self.controller.initialize(config, columns)
            self.start_btn.config(text="Stop Generation")
            self.export_btn.config(state="disabled")
            self.analyze_btn.config(state="disabled")
            self.progress_var.set(0)
            self.log_text.config(state="normal")
            self.log_text.delete(1.0, "end")
            self.log_text.config(state="disabled")
            
            self.controller.start_generation_thread()
            
        except Exception as e:
            messagebox.showerror("Configuration Error", str(e))

    def export_data(self, format_type: str):
        if format_type == "csv":
            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            callback = self.controller.export_csv
        elif format_type == "json":
            filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
            callback = self.controller.export_json
        elif format_type == "sql":
            filename = filedialog.asksaveasfilename(defaultextension=".sql", filetypes=[("SQL Files", "*.sql")])
            callback = self.controller.export_sql
        elif format_type == "pdf_report":
            filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
            callback = self.controller.export_pdf_report
        elif format_type == "pdf_narrative":
            filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
            callback = self.controller.export_narrative_pdf
        else:
            return

        if filename:
            callback(filename)
            messagebox.showinfo("Export", f"Saved to {filename}")

    def save_config(self):
        try:
            config_data = {
                "model_id": self.model_id_var.get(),
                "provider": self.provider_var.get(),
                "api_key": self.api_key_var.get() if self.provider_var.get() != AIProvider.LM_STUDIO.value else None,
                "azure_endpoint": self.azure_endpoint_var.get() if self.provider_var.get() == AIProvider.AZURE_OPENAI.value else None,
                "azure_deployment": self.azure_deployment_var.get() if self.provider_var.get() == AIProvider.AZURE_OPENAI.value else None,
                "num_rows": self.rows_var.get(),
                "similarity_threshold": self.sim_var.get(),
                "max_retries": self.retries_var.get(),
                "columns": [col.get_definition().model_dump() for col in self.column_widgets]
            }
            
            # Ensure templates dir exists
            templates_dir = os.path.join(os.getcwd(), "templates")
            os.makedirs(templates_dir, exist_ok=True)

            filename = filedialog.asksaveasfilename(
                initialdir=templates_dir,
                defaultextension=".json", 
                filetypes=[("JSON Files", "*.json")]
            )
            
            if filename:
                import json
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4)
                messagebox.showinfo("Success", f"Configuration saved to {filename}")
                
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def load_config(self):
        try:
            # Ensure templates dir exists (just in case)
            templates_dir = os.path.join(os.getcwd(), "templates")
            os.makedirs(templates_dir, exist_ok=True)

            filename = filedialog.askopenfilename(
                initialdir=templates_dir,
                filetypes=[("JSON Files", "*.json")]
            )
            
            if not filename:
                return
                
            import json
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Application Logic
            # 1. Clear existing columns
            for widget in self.column_widgets[:]:
                self._remove_column(widget)
            
            # 2. Set Config
            if "model_id" in data:
                self.model_combo.set(data["model_id"])
            if "provider" in data:
                self.provider_var.set(data["provider"])
                self._on_provider_change(None)  # Trigger UI update
            if "api_key" in data and data["api_key"]:
                self.api_key_var.set(data["api_key"])
            if "azure_endpoint" in data and data["azure_endpoint"]:
                self.azure_endpoint_var.set(data["azure_endpoint"])
            if "azure_deployment" in data and data["azure_deployment"]:
                self.azure_deployment_var.set(data["azure_deployment"])
            if "num_rows" in data:
                self.rows_var.set(data["num_rows"])
            if "similarity_threshold" in data:
                self.sim_var.set(data["similarity_threshold"])
            if "max_retries" in data:
                self.retries_var.set(data["max_retries"])
                
            # 3. Add Columns
            if "columns" in data:
                for col_data in data["columns"]:
                    # Handle Pydantic model reconstruction
                    col_def = ColumnDefinition(**col_data)
                    self._add_column(col_def)
            
            messagebox.showinfo("Success", "Configuration loaded!")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load config: {e}")

    def import_data(self):
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("Data Files", "*.csv *.json")]
            )
            if not filename:
                return

            import pandas as pd
            import json
            
            # Load Data
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
            else:
                df = pd.read_json(filename)
                
            # Convert to list of dicts for the controller
            # orient='records' gives [{col: val}, ...]
            self.imported_data = df.to_dict(orient='records')
            
            if not self.imported_data:
                messagebox.showerror("Error", "File is empty.")
                return

            # Infer Schema
            # Clear existing columns
            for widget in self.column_widgets[:]:
                self._remove_column(widget)
                
            # Create columns from dataframe
            for col_name in df.columns:
                # Infer type roughly
                dtype = df[col_name].dtype
                col_type = ColumnType.SHORT_TEXT
                if 'int' in str(dtype) or 'float' in str(dtype):
                    col_type = ColumnType.NUMERIC
                    
                # For imported data, we still assume Short Text for most things to be safe, 
                # or we could make a new "IMPORTED" type? 
                # For now, let's just make them Short Text, but maybe mark them?
                # The Controller skips generation if value exists, strictly based on name matching.
                
                col_def = ColumnDefinition(
                    name=col_name,
                    type=col_type,
                    prompt_instruction="(Imported)"
                )
                self._add_column(col_def)
                
            # Lock Row Count
            count = len(self.imported_data)
            self.rows_var.set(count)
            # We assume user wants to enrich ALL rows, so we enforce num_rows = imported rows
            
            messagebox.showinfo("Success", f"Imported {count} rows. Added columns to schema.\nAdd NEW columns to enrich this data.")
            
        except Exception as e:
            messagebox.showerror("Import Error", str(e))
        except Exception as e:
            messagebox.showerror("Import Error", str(e))
            self.imported_data = None

    def _on_analyze(self):
        metrics = self.controller.analyze_quality()
        if not metrics:
            messagebox.showinfo("Analysis", "No generated data to analyze.")
            return
        
        # Create Toplevel window for results
        top = tk.Toplevel(self)
        top.title("Data Quality Analysis")
        top.geometry("600x400")
        
        txt = tk.Text(top, wrap="word", padx=10, pady=10)
        txt.pack(fill="both", expand=True)
        
        msg = "=== DATA QUALITY REPORT ===\n\n"
        for col, data in metrics.items():
            msg += f"COLUMN: {col}\n"
            msg += f"{'-'*30}\n"
            msg += f"  • Diversity Score: {data.get('diversity_score', 0):.1%}\n"
            msg += f"  • Null Count:      {data.get('null_count', 0)}\n"
            msg += f"  • Top Frequent Values:\n"
            
            frequent = data.get('top_frequent', {})
            for val, count in frequent.items():
                msg += f"      - {val}: {count}\n"
            msg += "\n"
            
        txt.insert("end", msg)
        txt.config(state="disabled")

    def _open_help(self):
        from .help_window import HelpWindow
        HelpWindow(self)


