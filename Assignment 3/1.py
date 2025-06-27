# Rewriting the same content in a cleaner PDF using FPDF with basic formatting for structure
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Set title
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Report: Classification Using Raw vs Pre-Processed Dataset", ln=True)

# Set body font
pdf.set_font("Arial", '', 11)

sections = [
    ("Objective", 
     "Build a machine learning model to classify customers based on their demographic and behavioral data. "
     "Two approaches were tested:\n"
     "1. Using the raw dataset with minimal processing\n"
     "2. Using a fully pre-processed version of the dataset"),
    
    ("1. Classification Using Raw Dataset", 
     "Steps:\n"
     "- Dropped rows with missing values.\n"
     "- Dropped the 'Legacy_Customer_ID' and 'Customer_Feedback' columns.\n"
     "- Categorical columns were encoded using Ordinal Encoding.\n"
     "- Data was split into training and test sets (70% training, 30% testing), using stratification.\n"
     "- A Random Forest Classifier was trained.\n"
     "- Evaluated with accuracy, precision, recall, F1 score, and ROC AUC.\n\n"
     "Notes:\n"
     "- Simple and fast to implement.\n"
     "- Dropping rows can lead to information loss.\n"
     "- Ordinal encoding may not be suitable if there is no natural order.\n"
     "- Lacks feature scaling and advanced handling."),
    
    ("2. Classification Using Pre-Processed Dataset", 
     "Motivation:\n"
     "Most real-world datasets have missing values and varying data types. Preprocessing is needed.\n\n"
     "Preprocessing Strategy:\n"
     "- Numerical: impute with mean, scale with StandardScaler\n"
     "- Categorical: impute with most frequent, One-Hot Encode\n"
     "- Drop ID and text columns\n\n"
     "Implementation:\n"
     "- Used ColumnTransformer for different types\n"
     "- Built a pipeline to include preprocessing and classifier\n"
     "- Used Random Forest again\n\n"
     "Advantages:\n"
     "- Retains data with imputation\n"
     "- One-hot encoding properly handles categories\n"
     "- Scaling improves performance\n"
     "- Suitable for production"),
    
    ("Conclusion", 
     "Raw Dataset:\n"
     "  Pros:\n"
     "    - Simple, fast, and easy to implement\n"
     "  Cons:\n"
     "    - Discards data with missing values\n"
     "    - May misrepresent categorical data\n"
     "    - Lower accuracy compared to a pre-processed model\n\n"
     "Pre-Processed Data:\n"
     "  Pros:\n"
     "    - More accurate and reliable results\n"
     "    - Handles missing values and categorical variables properly\n"
     "    - Suitable for production or real-world scenarios\n"
     "  Cons:\n"
     "    - Requires more setup and understanding of data preprocessing techniques"),
]


# Add each section to PDF
for title, content in sections:
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, content)
    pdf.ln(2)

# Save the final PDF
final_pdf_path = "classification_report_clean.pdf"

pdf.output(final_pdf_path)

