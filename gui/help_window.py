import tkinter as tk
from tkinter import ttk
import webbrowser

class HelpWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Synthetic Data Generator - Documentation")
        self.geometry("700x600")
        
        # Create Notebook (Tabs)
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Basics
        self._create_basics_tab(notebook)
        
        # Tab 2: Column Types
        self._create_types_tab(notebook)
        
        # Tab 3: Advanced
        # Tab 3: Advanced
        self._create_advanced_tab(notebook)
        
        # Tab 4: Examples
        self._create_examples_tab(notebook)
        
        # Close Button
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=10)

    def _create_basics_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="Basics")
        
        text = (
            "Welcome to the Synthetic Data Generator!\n\n"
            "This tool helps you create fake data for testing software, filling databases, "
            "or conducting research. Instead of just random gibberish, it uses Artificial Intelligence (AI) "
            "to generate realistic content.\n\n"
            "How it works:\n"
            "1. Define Columns: Choose what kind of data you want (names, reviews, scores).\n"
            "2. Configure Model: Select your local AI model.\n"
            "3. Click Start: The app will generate row by row.\n\n"
            "The data is checked for uniqueness so you don't get duplicates."
        )
        
        lbl = tk.Label(frame, text=text, justify="left", wraplength=600, font=("Segoe UI", 10))
        lbl.pack(anchor="nw")

    def _create_types_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="Column Types")
        
        # Scrollable text area for types
        txt = tk.Text(frame, wrap="word", width=80, height=25, font=("Segoe UI", 10), bg="#f0f0f0", relief="flat")
        txt.pack(fill="both", expand=True)
        
        content = [
            ("Short Text", "Generates creative, short answers using AI. Good for: Cities, Colors, Job Titles."),
            ("Long Text", "Generates paragraph-length text. The app checks these for 'semantic duplication' to ensure they are unique. Good for: Reviews, Bios, Descriptions."),
            ("Numeric", "Generates a number. You can enforce constraints (e.g. 1 to 10)."),
            ("Categorical", "Picks from a list of options. Good for: Status (Active/Inactive), T-Shirt Size (S/M/L)."),
            ("Boolean", "True or False."),
            ("Auto Increment", "A simple counter (1, 2, 3...). Used for IDs."),
            ("Faker / Deterministic", "⚡ FASTEST OPTION. Uses standard libraries instead of AI. Use this for Names, Emails, Dates, Addresses.")
        ]
        
        for title, desc in content:
            txt.insert("end", f"• {title}\n", "bold")
            txt.insert("end", f"  {desc}\n\n")
            
        txt.tag_config("bold", font=("Segoe UI", 10, "bold"))
        txt.config(state="disabled")

    def _create_advanced_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="Advanced")
        
        txt = tk.Text(frame, wrap="word", width=80, height=25, font=("Segoe UI", 10), bg="#f0f0f0", relief="flat")
        txt.pack(fill="both", expand=True)
        
        sections = [
            ("Allow Duplicates", 
             "By default, the app is strict and rejects duplicates. "
             "Check this box if you WANT repeated values (e.g., for a 'Category' column where multiple rows can be 'Admin')."),
            
            ("Magic Generator", 
             "Don't want to build columns manually? Type a description like "
             "'A dataset of medieval potions with effects and rarity' into the Magic Generator box and click 'Auto-Generate'."),
             
            ("Logic Constraints", 
             "Enforce rules between columns using Python syntax OR natural language.\n"
             "• after @[Start Date]\n"
             "• greater than 100\n"
             "• shorter than 20\n"
             "• this != @[Other Column]"),

            ("Regex Validation",
             "Ensure the output matches a specific pattern.\n"
             "You can use standard Regex OR shortcuts:\n"
             "• email\n"
             "• phone\n"
             "• zip / postcode\n"
             "• date (YYYY-MM-DD)\n"
             "• ipv4"),

            ("Similarity Threshold",
             "Controls how strict the 'Semantic Uniqueness' check is for Long Text.\n"
             "• 0.0: No check (allow everything)\n"
             "• 0.85: Default\n"
             "• 1.0: Exact matches only\n"
             "Lower this if the AI is struggling to find unique values."),

            ("Context Linking (@[column])",
             "You can reference values from previous columns in your Prompt to make them related.\n"
             "Example:\n"
             "• Column 1 'City': Generate a city name.\n"
             "• Column 2 'Weather': What is the weather like in @[City]?\n\n"
             "The app will automatically generate 'City' first, then inject that value into the prompt for 'Weather'."),

            ("Exporting", 
             "You can save your data as CSV (Excel), JSON (Web), or SQL (Database). "
             "You can also generate a PDF Quality Report to see how diverse your data is.")
        ]
        
        for title, desc in sections:
            txt.insert("end", f"• {title}\n", "bold")
            txt.insert("end", f"  {desc}\n\n")
            
        txt.tag_config("bold", font=("Segoe UI", 10, "bold"))
        txt.config(state="disabled")

    def _create_examples_tab(self, notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="Examples")
        
        txt = tk.Text(frame, wrap="word", width=80, height=25, font=("Segoe UI", 10), bg="#f0f0f0", relief="flat")
        txt.pack(fill="both", expand=True)
        
        examples = [
            ("1. Product Inventory (Logic & Math)", 
             "• Price (Numeric): Min=10, Max=1000\n"
             "• Discount Price (Numeric): Logic = 'this < @[Price]'\n"
             "• Stock (Numeric): Min=0, Max=500\n"
             "• Is Available (Boolean): -"),
            
            ("2. Shipping Logistics (Dates)", 
             "• Order Date (Faker): Date between -30d and today\n"
             "• Delivery Date (Faker): Date between today and +30d\n"
             "• Valid Delivery (Boolean): Logic = '@[Delivery Date] after @[Order Date]'"),

            ("3. User Verification (Regex)", 
             "• Username (Short Text): -\n"
             "• Phone (Short Text): Regex = '^\\d{3}-\\d{3}-\\d{4}$' (e.g. 123-456-7890)\n"
             "• Zip Code (Short Text): Regex = '^\\d{5}$'"),

            ("4. Travel Itinerary (Context Linking)", 
             "• Destination (Short Text): 'A popular tourist city'\n"
             "• Activity (Short Text): 'A fun activity to do in @[Destination]'\n"
             "• Review (Long Text): 'A positive review of doing @[Activity] in @[Destination]'")
        ]
        
        for title, desc in examples:
            txt.insert("end", f"{title}\n", "bold")
            txt.insert("end", f"{desc}\n\n")
            
        txt.tag_config("bold", font=("Segoe UI", 10, "bold"))
        txt.config(state="disabled")


