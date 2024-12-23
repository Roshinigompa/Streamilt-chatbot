import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import speech_recognition as sr
import base64


# Load GPT-2 model and tokenizer
model_name = "gpt2"  # You can change this to a larger GPT model if needed
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad token for GPT-2 (GPT-2 doesn't have a pad token by default)
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as padding token

# Function to generate response using GPT-2 model
def generate_gpt2_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Ensure input_ids is a tensor before creating attention mask
    input_ids = torch.tensor(input_ids) if not isinstance(input_ids, torch.Tensor) else input_ids
    
    # Create the attention mask, where 1 indicates non-padding and 0 indicates padding
    attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Fix for padding
    
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=150, do_sample=True, temperature=0.85)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Function to handle custom responses based on keywords
# Function to handle custom responses based on keywords
def generate_custom_response(user_input):
    user_input_lower = user_input.lower()

    # Define multiple keywords for background (or any other category)
    background_keywords = ["background", "about you", "who are you", "tell me about yourself", "introduce yourself"]
    approach_keywords = ["approach", "method", "strategy"]
    learning_engineering_keywords = ["learning engineering", "educational engineering", "learning strategies"]
    skills_keywords = ["skills", "expertise", "capabilities"]
    vision_keywords= ["vision","contibute","contribution","future"]
    projects_keywords = ["projects", "project list"]
    recent_projects_keywords = ["recent projects", "latest projects", "current projects"]
    ai_keywords=["ai tools"]
    Collab_keywords=["collaboration"]
    future_keywords=["future","goal"]



    # Check if any background keyword is in the user's input
    if any(keyword in user_input_lower for keyword in background_keywords):
        return "I’m Hyma Roshini Gompa. I am currently pursuing a Master's in Data Science at the University of Maryland Baltimore County (UMBC), where I maintain a 3.95 GPA. I previously earned a Bachelor's in Computer Science from Lovely Professional University with a GPA of 3.9. My experience spans across AI and data science, where I have worked with machine learning frameworks like Python, TensorFlow, and Scikit-learn. I've developed predictive models for survival analysis and worked on end-to-end machine learning workflows with MLflow and AWS CI/CD. I also have significant experience in business analysis, having worked as a Business Analyst Intern at Bizinc, where I enhanced cross-team collaboration and designed SQL-based performance dashboards to streamline reporting."
    elif any(keyword in user_input_lower for keyword in approach_keywords):
        return "My approach to the AI Development Intern role would focus on collaboration, innovation, and user-centered design. First, I would work closely with the FutureMakers team to understand the unique needs of educators using Readyness. I would develop AI-driven features that dynamically generate tailored materials and provide suggestions for aligning projects with standards like NGSS and CASEL. My method would involve utilizing machine learning models to process existing data and improve the educator experience by simplifying lesson planning. I would also iterate based on user feedback to ensure the AI tools are intuitive and effective in real classroom settings."
    elif any(keyword in user_input_lower for keyword in learning_engineering_keywords):
        return "Learning engineering is the application of research-based strategies to design and optimize educational experiences, particularly for adult learners. In my projects, I've always prioritized usability and user engagement, especially when designing systems that require user interaction. For example, in my past work, I focused on creating intuitive data dashboards that improved decision-making by making complex information more accessible. For this role, I would apply principles of learning engineering to ensure that the AI tools within Readyness are tailored to educators' needs, empowering them to build their confidence. I would also focus on adult learning strategies by considering how educators engage with content and how AI can support them in achieving mastery with minimal effort."
    elif any(keyword in user_input_lower for keyword in vision_keywords):
        return "I believe the future of AI in education is about creating personalized, adaptive learning environments that cater to the unique needs of each learner, while empowering educators to make data-driven decisions. As AI continues to evolve, it will play a critical role in reducing administrative burdens, providing real-time insights, and enabling more inclusive teaching practices. I see myself contributing to this vision by developing AI tools like Readyness that focus on teacher empowerment and student-centered learning. I would focus on improving AI models that dynamically adjust content and provide educators with resources tailored to both their teaching styles and students’ needs."
    elif any(keyword in user_input_lower for keyword in skills_keywords):
        return "I have a strong background in data science and AI/ML, including experience with Python, TensorFlow, and OpenAI GPT models. My expertise extends to SQL, machine learning frameworks like Scikit-learn, and data visualization tools like Tableau and Power BI. I am also proficient in using DevOps tools such as Docker, Jenkins, and Kubernetes for model deployment. I have hands-on experience with creating APIs and integrating them with AI models to build intuitive user interfaces. Additionally, I have a solid understanding of tools like Airtable and Zapier, which would help streamline workflows and automate processes within the Readyness platform."
    elif any(keyword in user_input_lower for keyword in projects_keywords):
        return """Here are my recent projects:
1. **Anomaly Detection Using Robust Graphical Lasso (RGlasso)**:
   Implemented a novel approach for anomaly detection in high-dimensional datasets using RPCA and RGlasso, demonstrating improved precision and robustness.
2. **Breast Cancer Survival Prediction**:
   Compared machine learning models with traditional survival analysis methods on the METABRIC dataset, leveraging RandomForestClassifier for predictions.
    Here are some of my notable projects:
1. **Survival Analysis for Breast Cancer Prediction (Aug 2024 - Dec 2024)**:
   Developed survival models (Kaplan-Meier, Cox PH, Random Survival Forests) for breast cancer prognosis using Python and MATLAB, improving predictive accuracy by 15% on the METABRIC dataset.
2. **End-to-End Machine Learning Workflow with MLflow and AWS CICD (May 2024 – Jun 2024)**:
   Integrated MLflow for experiment tracking and deployed containerized models using Docker and AWS EC2. Automated CI/CD with GitHub Actions, reducing deployment time by 40%.
3. **Real-Time Voting Analysis System (Apr 2024 - May 2024)**:
   Built a scalable, real-time voting system with Python, Kafka, and Spark Streaming, capable of processing 500,000+ records/minute.
4. **Tracking-the-NASA-Satellite (Jan 2024 – Feb 2024)**:
   Built a real-time NASA satellite tracker using Python to process and display live data from the NASA API."""
    elif any(keyword in user_input_lower for keyword in recent_projects_keywords):
        return """Here are my recent projects:
1. **Anomaly Detection Using Robust Graphical Lasso (RGlasso)**:
   Implemented a novel approach for anomaly detection in high-dimensional datasets using RPCA and RGlasso, demonstrating improved precision and robustness.
2. **Breast Cancer Survival Prediction**:
   Compared machine learning models with traditional survival analysis methods on the METABRIC dataset, leveraging RandomForestClassifier for predictions."""
    elif any(keyword in user_input_lower for keyword in ai_keywords):
        return (
            "I am proficient in various AI tools and libraries, including TensorFlow, Scikit-learn, PyTorch (for non-deep learning tasks), "
            "OpenAI GPT models, and MLflow for managing machine learning experiments. I am also experienced in cloud-based tools like AWS "
            "for deployment and scaling, and I have used Docker and Jenkins for continuous integration and deployment. These tools have "
            "enabled me to build scalable AI applications and deploy models effectively in production environments."
        )
    elif any(keyword in user_input_lower for keyword in Collab_keywords):
        return (
            "Collaboration is a core value in my work. For example, at Bizinc, I collaborated closely with product managers, engineers, "
            "and stakeholders to design solutions that met the business’s needs. I’ve also worked on cross-functional teams during my time "
            "at Josh Technology Group, where we built predictive models that had a significant impact on decision-making processes. "
            "In any role, I focus on communication, knowledge sharing, and delivering value through teamwork."
        )
          
    elif any(keyword in user_input_lower for keyword in future_keywords):
        return (
            "In the future, I aim to work on more advanced AI systems that can provide personalized, data-driven solutions in sectors "
            "like education, healthcare, and business. I am particularly interested in expanding my expertise in natural language processing "
            "and reinforcement learning. Additionally, I aspire to take on leadership roles where I can drive AI-driven innovations and "
            "help organizations implement AI strategies that solve real-world problems effectively."
        )

    else:
        return "The command does not match any known responses. Please change the command and try again."


# Function to create chatbot interaction
# Function for speech-to-text input
def speech_to_text():
    recognizer = sr.Recognizer()
    container = st.empty()  # Create an empty container for dynamic updates

    try:
        # Check if a default microphone is available
        with sr.Microphone() as source:
            container.info("Listening... Please speak now!")  # Display the prompt
            try:
                # Listen to the audio input
                audio = recognizer.listen(source, timeout=5)
                spoken_text = recognizer.recognize_google(audio)
                container.empty()  # Clear the prompt after recognition
                return spoken_text
            except sr.UnknownValueError:
                container.error("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                container.error(f"Could not request results; {e}")
            finally:
                container.empty()  # Ensure the prompt is cleared in all cases
    except OSError:
        # Handle the case where no microphone is available
        container.error("No microphone input device available. Please ensure a microphone is connected.")
        return "No microphone available."
    
    return ""

# Chatbot interaction with UI enhancements
def chatbot_interaction():
    st.title("This is Hyma Roshini Gompa, Know More About Me")
    st.write("Explore my qualifications, projects, and approach to AI-driven solutions.")

    # User input field
    user_input = st.text_input("Ask me about my background, skills, or projects:")

       # Buttons for predefined queries
    col0, col1, col2, col3, col4 = st.columns(5)

    

    with col0:
        if st.button("📄 Resume"):
            try:
                # Cloud-hosted Google Drive link for the resume
                drive_link = "https://drive.google.com/file/d/15b9T2H3QeYCDcUiTvStCeMn3jh9lgZWx/preview"  # Replace with your actual link
                download_link = "https://drive.google.com/uc?id=15b9T2H3QeYCDcUiTvStCeMn3jh9lgZWx&export=download"  # Direct download link
                st.markdown(f'<iframe src="{drive_link}" width="700" height="1000"></iframe>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <a href="{download_link}" target="_blank" style="
                    display: inline-block;
                    font-size: 16px;
                    font-weight: bold;
                    color: white;
                    background-color: #4CAF50;
                    padding: 10px 15px;
                    border-radius: 5px;
                    text-decoration: none;
                    text-align: center;
                    ">Download Resume</a>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"An error occurred while displaying the resume: {e}")

    with col1:
        if st.button("Background"):
            user_input = "background"
    with col2:
        if st.button("Skills"):
            user_input = "skills"
    with col3:
        if st.button("Projects"):
            user_input = "projects"
    with col4:
        if st.button("🎤 Speak"): # Speech-to-text button
            user_input = speech_to_text()
    


    if user_input:
        custom_response = generate_custom_response(user_input)
        if custom_response:
            st.write(f"Chatbot: {custom_response}")
        else:
            gpt2_response = generate_gpt2_response(user_input)
            st.write(f"Chatbot: {gpt2_response}")

    # Portfolio and project visuals
    with st.expander("📂 Portfolio and Certificates"):
        st.write("Visit my [LinkedIn](https://www.linkedin.com/in/gompa-hyma/) or [GitHub](https://github.com/Roshinigompa) profiles.")
        

if __name__ == "__main__":
    chatbot_interaction()
    