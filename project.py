import streamlit as st
import pdfplumber
import re
import pickle
import google.generativeai as genai


# Load pre-trained vectorizers, selector and models
with open("resume_screening_vectorizer_rf.pkl", "rb") as f:
    vectorizer_rf = pickle.load(f)
with open("resume_screening_vectorizer_nb.pkl", "rb") as f:
    vectorizer_nb = pickle.load(f)
with open("resume_selector.pkl", "rb") as f:
    selector = pickle.load(f)
with open("resume_screening_rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("resume_screening_nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

# Configure Gemini AI
GENAI_API_KEY = "AIzaSyADEjVMNytfF5GL_i8fgUtvdQ2nqnZ5TEE"
genai.configure(api_key=GENAI_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not Found"

def extract_phone(text):
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else "Not Found"

def extract_name_using_regex(text):
    name_patterns = [
        r"^[A-Z\s]+\n",
        r"[A-Z][a-z]+\s[A-Z][a-z]+",
        r"[A-Z][a-z]+\s[A-Z]\.[A-Z][a-z]+",
        r"[A-Z][a-z]+\s[A-Z][a-z]+\s[A-Z][a-z]+",
    ]
    for pattern in name_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
    return "Not Found"
def extract_skills(text):
    predefined_skills =[
    "UX Design", "Wireframing", "Prototyping", "User Research", "User Interface Design", "Adobe XD", "Figma", 
    "Interaction Design", "Information Architecture", "Design Thinking", "User Flow", "Sketch", "Usability Testing",
    "Networking", "TCP/IP", "DNS", "DHCP", "Routing", "Switching", "Firewalls", "VPN", "Network Security", 
    "Server Administration", "LAN/WAN", "Network Monitoring", "IP Addressing", "Load Balancing", "Network Troubleshooting",
    "UI Design", "HTML", "CSS", "Bootstrap", "Web Design", "Responsive Design", "Mobile-First Design",
    "Social Media Strategy", "Content Creation", "SEO", "Analytics", 
    "Google Analytics", "Brand Monitoring", "Social Media Marketing", "Community Engagement", "Influencer Marketing", 
    "Copywriting", "Social Media Ads", "UX Research", "User-Centered Design", "Persona Development", "Usability Research",
    "Customer Journey Mapping", "Procurement", "Vendor Management", "Supply Chain Management", "Contract Management", 
    "Negotiation", "Inventory Management", "Purchase Orders", "Excel", "SAP", "Sourcing", "Risk Management", 
    "Market Research", "Budgeting", "Social Media Analytics", "Data Analysis", "Facebook Insights", "Twitter Analytics", 
    "Instagram Insights", "Content Strategy", "Campaign Reporting", "Social Media Metrics", "Conversion Tracking", 
    "SEMRush", "Social Listening", "Quality Testing", "Bug Tracking", "Test Automation", "Selenium", "JIRA", "Manual Testing", 
    "SQL", "Regression Testing", "Performance Testing", "Test Cases", "UAT", "Load Testing", "Agile Methodology", 
    "Continuous Integration", "On-Page SEO", "Off-Page SEO", "Keyword Research", "Link Building", "Content Optimization", 
    "Google Search Console", "SEO Audits", "SEO Tools", "Technical SEO", "Website Analysis", "Backlinking", 
    "Office Management", "Scheduling", "Travel Coordination", "Calendar Management", "Communication", "MS Office", 
    "Organization", "Data Entry", "Customer Service", "Time Management", "Filing", "Documentation", "Event Coordination", 
    "Correspondence", "Database Management", "MySQL", "Oracle", "Performance Tuning", "Backups", "Security", 
    "Data Integrity", "Data Modeling", "Query Optimization", "Database Design", "Cloud Databases", "Database Replication", 
    "Data Warehousing", "NoSQL", "ERP Systems", "Market Analysis", "Supplier Relationship Management", "Strategic Sourcing", 
    "Logistics", "Python", "R", "Tableau", "Power BI", "Data Cleaning", "Machine Learning", "Predictive Analytics", 
    "Statistical Analysis", "Data Mining", "Big Data", "Data Reporting", "Java", "Node.js", "C++", "Ruby", "RESTful APIs", 
    "Databases", "Microservices", "Server-Side Scripting", "Git", "Cloud Computing", "Docker", "Kubernetes", "AWS", 
    "Spring Boot", "Laravel", "Express.js", "Demand Forecasting", "Statistical Forecasting", "Sales Forecasting", 
    "Sales Data", "Forecasting Models", "Supply Chain Optimization", "Web Development", "React", "Angular", "Vue.js", 
    "jQuery", "AJAX", "UI/UX Design", "Responsive Web Design", "Cross-Browser Compatibility", "TypeScript", "Sass", 
    "Web Performance Optimization", "Customer Support", "CRM", "Client Relations", "Customer Retention", "Problem-Solving", 
    "Product Training", "Onboarding", "Customer Satisfaction", "Feedback Collection", "Upselling", "Account Management", 
    "Financial Planning", "Retirement Planning", "Investment Strategies", "Tax Planning", "Portfolio Management", 
    "Estate Planning", "Social Security", "Annuities", "Asset Allocation", "Financial Models", "Client Relationship Management",
    "Sales", "Client Relationship", "B2B Sales", "Lead Generation", "Business Development", "Cold Calling", "Proposal Writing", 
    "Customer Engagement", "Sales Reporting", "Territory Management", "Pipeline Management", "Customer Segmentation", 
    "Product Knowledge", "Visual Design", "Network Security", "Penetration Testing", "Intrusion Detection", "Encryption", 
    "Cybersecurity", "Vulnerability Assessment", "SIEM", "Security Audits", "Security Policies", "Security Tools", 
    "Incident Response", "Legal Research", "Document Preparation", "Case Management", "Contract Review", "Legal Writing", 
    "Litigation Support", "Court Filings", "Compliance", "Legal Analysis", "Evidence Handling", "Law Office Management", 
    "Legal Drafting", "Law Practice Software", "Employee Training", "Curriculum Design", "Training Sessions", 
    "Learning & Development", "HR", "Instructional Design", "Employee Development", "Performance Evaluation", 
    "Training Materials", "Online Learning Platforms", "Event Planning", "Client Communication", "Event Marketing", 
    "Event Promotion", "Venue Selection", "Event Registration", "Event Execution", "Event Budgeting", "Email Management", 
    "Personal Support", "Sustainable Design", "Green Building", "LEED Certification", "Energy Efficiency", 
    "Environmental Impact", "Renewable Energy", "Waste Management", "Water Conservation", "Building Codes", 
    "Sustainable Materials", "Environmental Policy", "Carbon Footprint", "Deep Learning", "TensorFlow", "Feature Engineering", 
    "Predictive Modeling", "NLP", "Server Management", "Linux", "Windows", "Virtualization", "Backup Solutions", 
    "System Monitoring", "Active Directory", "VMware", "Troubleshooting", "Automation", "Systems Configuration"
]
    return [skill for skill in predefined_skills if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE)]
def get_resume_enhancement(job_role, resume_text):
    """Use Gemini AI to generate resume enhancement suggestions."""
    prompt = (
        f"Analyze the following resume for a '{job_role}' role and provide specific, actionable enhancement suggestions. "
        f"Ensure your response is structured professionally with clear formatting.\n\n"
        f"## Resume Analysis and Enhancement Suggestions for a {job_role} Role\n\n"
        f"### Formatting & Clarity Improvements:\n"
        f"- Present contact information cleanly.\n"
        f"- Use consistent formatting for section headings (bold, all caps, etc.).\n"
        f"- Maintain consistent date formatting (e.g., May 2023 - July 2023 or 05/2023 - 07/2023).\n"
        f"- Use bullet points effectively with strong action verbs and quantified achievements.\n"
        f"- Arrange experience and education in reverse chronological order.\n\n"
        f"### Skills Enhancement:\n"
        f"- Identify missing or outdated technical skills relevant to the role.\n"
        f"- Suggest advanced skills and certifications that are in demand.\n"
        f"- Recommend cloud platforms, MLOps tools, and big data technologies.\n\n"
        f"### Experience Refinement:\n"
        f"- Provide quantifiable achievements for work experience.\n"
        f"- Recommend moving certifications to a separate section.\n"
        f"- Clarify project contributions and impact on business outcomes.\n\n"
        f"### Project Recommendations:\n"
        f"- Expand project descriptions with datasets, evaluation metrics, and technologies used.\n"
        f"- Include links to GitHub repositories or deployed applications.\n"
        f"- Suggest adding industry-relevant projects such as recommendation systems or time series forecasting.\n\n"
        f"### Education:\n"
        f"- Include expected graduation date and GPA if above average.\n"
        f"- List relevant coursework like Machine Learning, Deep Learning, and Data Structures.\n\n"
        f"### Professional Summary:\n"
        f"- Make it concise and impactful.\n"
        f"- Highlight key skills, achievements, and career goals.\n\n"
        f"Resume Text:\n{resume_text}"
    )
    model = genai.GenerativeModel("gemini-2.5-flash") 
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit UI
st.title("AI-Powered Resume Screening and Enhancement")

# Sidebar for navigation
option = st.sidebar.radio("Select an Option", ["Resume Screening", "Resume Enhancement"])

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if option == "Resume Screening":
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        if text:
            email = extract_email(text)
            phone = extract_phone(text)
            name = extract_name_using_regex(text)
            skills= extract_skills(text)
            X_rf_transformed = vectorizer_rf.transform([text])
            text_selected = selector.transform(X_rf_transformed)
            X_nb_transformed = vectorizer_nb.transform([text])
            rf_prediction = rf_model.predict(text_selected)[0]
            nb_prediction = nb_model.predict(X_nb_transformed)[0]
            st.write(f"**Recommended Profession:** {rf_prediction}")
            st.write(f"**Recommended Job Role:** {nb_prediction.upper()}")
            st.subheader("Screening Results")
            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
            st.write(f"**Phone:** {phone}")
            
            st.markdown("**Skills:**")
            st.markdown("<br>".join([f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;•&nbsp;{skill}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" for skill in skills]), unsafe_allow_html=True)


    
        else:
            st.error("Could not extract text from the uploaded PDF.")

elif option == "Resume Enhancement":
    job_role = st.text_input("Enter Job Role")
    if st.button("Enhance Resume"):
        if uploaded_file and job_role:
            with st.spinner("Processing..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                if resume_text:
                    enhancements = get_resume_enhancement(job_role, resume_text)
                    st.subheader("Enhancement Suggestions")
                    st.write(enhancements)
                else:
                    st.error("Could not extract text from the uploaded PDF.")
        else:
            st.error("Please upload a PDF and enter a job role.")
