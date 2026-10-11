import streamlit as st
import requests
import json
import time
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from typing import Dict, List, Optional
import threading

# Set page config for professional appearance
st.set_page_config(
    page_title="D&D Campaign Forge",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional D&D theme
st.markdown("""
<style>
    :root {
        --primary-color: #8B0000;
        --secondary-color: #D4AF37;
        --background-color: #1a1a1a;
        --card-bg: #2d2d2d;
        --text-color: #e0e0e0;
        --border-radius: 12px;
        --shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    }
    
    .main-header {
        text-align: center;
        color: var(--secondary-color);
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 2rem;
        font-family: 'Cinzel', serif;
        letter-spacing: 1px;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1a0a0a 0%, #0a0a1a 100%);
        color: var(--text-color);
    }
    
    .stSidebar {
        background-color: rgba(26, 10, 10, 0.9);
        backdrop-filter: blur(10px);
        border-right: 1px solid var(--secondary-color);
    }
    
    .campaign-card {
        background: var(--card-bg);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        border: 1px solid #5d4037;
        transition: transform 0.3s ease;
    }
    
    .campaign-card:hover {
        transform: translateY(-5px);
        border-color: var(--secondary-color);
    }
    
    .section-title {
        color: var(--secondary-color);
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem;
        font-family: 'Cinzel', serif;
    }
    
    .stButton>button {
        background: linear-gradient(to right, var(--primary-color), #5d0000);
        color: white;
        border: 2px solid var(--secondary-color);
        border-radius: 30px;
        width: 100%;
        padding: 0.8rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(to right, #5d0000, var(--primary-color));
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.5);
    }
    
    .stSelectbox, .stTextInput, .stTextArea, .stSlider {
        background-color: #3a3a3a;
        border-radius: 8px;
        border: 1px solid #5d4037;
        color: var(--text-color);
    }
    
    .agent-message {
        background: rgba(45, 25, 15, 0.7);
        border-left: 4px solid var(--secondary-color);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 var(--border-radius) var(--border-radius) 0;
    }
    
    .user-message {
        background: rgba(25, 35, 55, 0.7);
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 var(--border-radius) var(--border-radius) 0;
    }
    
    .thinking-animation {
        display: inline-block;
        width: 20px;
        text-align: center;
        font-weight: bold;
        color: var(--secondary-color);
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        color: #888;
        font-style: italic;
        border-top: 1px solid #5d4037;
        margin-top: 2rem;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .pulse {
        animation: pulse 1.5s infinite;
    }
    
    .map-placeholder {
        background: linear-gradient(135deg, #2c0e0e 0%, #1a0a1a 100%);
        border: 2px solid var(--secondary-color);
        border-radius: var(--border-radius);
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--secondary-color);
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    
    .stat-block {
        display: flex;
        justify-content: space-around;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .stat-item {
        background: rgba(139, 0, 0, 0.2);
        border: 1px solid var(--secondary-color);
        border-radius: 8px;
        padding: 0.8rem;
        min-width: 120px;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--secondary-color);
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #aaa;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Initialize session state
if 'campaign_generated' not in st.session_state:
    st.session_state.campaign_generated = False
if 'campaign_data' not in st.session_state:
    st.session_state.campaign_data = None
if 'ollama_models' not in st.session_state:
    st.session_state.ollama_models = []
if 'agent_conversation' not in st.session_state:
    st.session_state.agent_conversation = []
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Function to get available Ollama models
@st.cache_data(ttl=300)
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return sorted(models)
        return []
    except Exception as e:
        st.sidebar.error(f"Error connecting to Ollama: {str(e)}")
        return []

# Function to initialize AutoGen agents
def initialize_agents(model_name):
    # Configure AutoGen to use Ollama
    config_list = [
        {
            'model': model_name.split(':')[0],  # Remove tag if present
            'base_url': 'http://localhost:11434/v1',
            'api_key': 'ollama',  # Required for Ollama's OpenAI compatibility
            'temperature': 0.7,
        }
    ]
    
    # Campaign Master Agent - Oversees the entire process
    campaign_master = AssistantAgent(
        name="CampaignMaster",
        system_message="""You are an expert D&D campaign designer with decades of experience. Your role is to coordinate the entire campaign creation process, ensuring all elements are cohesive and engaging. You will delegate tasks to specialized agents and synthesize their contributions into a complete campaign.""",
        llm_config={"config_list": config_list},
    )
    
    # Lore Weaver Agent - Creates world-building elements
    lore_weaver = AssistantAgent(
        name="LoreWeaver",
        system_message="""You are a master world-builder specializing in rich D&D lore. Create detailed settings, history, factions, deities, and cultural elements. Focus on making the world feel alive with unique traditions, conflicts, and mysteries.""",
        llm_config={"config_list": config_list},
    )
    
    # Encounter Crafter Agent - Designs balanced encounters
    encounter_crafter = AssistantAgent(
        name="EncounterCrafter",
        system_message="""You are an expert encounter designer for D&D campaigns. Create balanced combat, social, and exploration encounters appropriate for the specified level range. Include tactical details, environmental factors, and meaningful consequences.""",
        llm_config={"config_list": config_list},
    )
    
    # NPC Architect Agent - Develops memorable characters
    npc_architect = AssistantAgent(
        name="NPCArchitect",
        system_message="""You are a character creation specialist for D&D campaigns. Design memorable NPCs with distinct personalities, motivations, secrets, and connections to the main plot. Include both allies and antagonists with depth and complexity.""",
        llm_config={"config_list": config_list},
    )
    
    # User Proxy Agent - Initiates and manages the conversation
    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    )
    
    return {
        "campaign_master": campaign_master,
        "lore_weaver": lore_weaver,
        "encounter_crafter": encounter_crafter,
        "npc_architect": npc_architect,
        "user_proxy": user_proxy
    }

# Function to generate campaign using AutoGen
def generate_campaign(agents, campaign_params):
    # Create task description from parameters
    task_description = f"""
    Create a complete D&D 5e campaign based on these specifications:
    
    Campaign Name: {campaign_params['name']}
    Theme: {campaign_params['theme']}
    Tone: {campaign_params['tone']}
    Character Levels: {campaign_params['level_range'][0]} to {campaign_params['level_range'][1]}
    Party Size: {campaign_params['party_size']}
    World Setting: {campaign_params['world_setting']}
    Special Requests: {campaign_params['special_requests'] if campaign_params['special_requests'] else 'None specified'}
    
    The campaign must include:
    1. An engaging hook to start the adventure
    2. Three major story arcs with clear objectives
    3. Detailed world-building including key locations, factions, and history
    4. 5-7 memorable NPCs with motivations and connections to the plot
    5. 8-10 balanced encounters (combat, social, exploration) appropriate for the level range
    6. A climactic finale with multiple possible outcomes
    7. Optional side quests and hidden secrets
    
    Structure the response in Markdown format with clear sections. Be creative and original!
    """
    
    # Register reply function to capture conversation
    conversation_log = []
    
    def log_reply(recipient, messages, sender, config):
        last_message = messages[-1]
        conversation_log.append({
            "sender": sender.name,
            "recipient": recipient.name,
            "content": last_message.get("content", ""),
            "timestamp": time.time()
        })
        st.session_state.agent_conversation = conversation_log
        return False, None
    
    # Register the reply function with all agents
    for agent_name, agent in agents.items():
        if agent_name != "user_proxy":
            agent.register_reply([agents["user_proxy"]], log_reply)
    
    # Start the conversation
    st.session_state.generation_complete = False
    with st.spinner("🧙‍♂️ Campaign Master is gathering the adventuring party..."):
        agents["user_proxy"].initiate_chat(
            agents["campaign_master"],
            message=task_description
        )
    
    st.session_state.generation_complete = True
    return conversation_log[-1]["content"] if conversation_log else "Campaign generation failed."

# Function to simulate campaign generation (for demo when Ollama not available)
def simulate_campaign_generation(campaign_params):
    # Simulate agent conversation
    conversation = [
        {"sender": "UserProxy", "recipient": "CampaignMaster", "content": "Create a D&D campaign with the given specifications", "timestamp": time.time()},
        {"sender": "CampaignMaster", "recipient": "LoreWeaver", "content": "LoreWeaver, please develop the world setting and major factions for this campaign.", "timestamp": time.time() + 1},
        {"sender": "LoreWeaver", "recipient": "CampaignMaster", "content": "I've created a rich world with ancient ruins, warring factions, and hidden mysteries. The world of Aethelgard is ready!", "timestamp": time.time() + 3},
        {"sender": "CampaignMaster", "recipient": "EncounterCrafter", "content": "EncounterCrafter, design balanced encounters for levels 1-5 in this setting.", "timestamp": time.time() + 5},
        {"sender": "EncounterCrafter", "recipient": "CampaignMaster", "content": "Completed: 8 encounters including a haunted forest, bandit ambush, and a dungeon crawl through ancient ruins.", "timestamp": time.time() + 8},
        {"sender": "CampaignMaster", "recipient": "NPCArchitect", "content": "NPCArchitect, create key NPCs that drive the story forward.", "timestamp": time.time() + 10},
        {"sender": "NPCArchitect", "recipient": "CampaignMaster", "content": "Developed 6 NPCs including a mysterious patron, a betrayed knight, and a cult leader with hidden motives.", "timestamp": time.time() + 13},
        {"sender": "CampaignMaster", "recipient": "UserProxy", "content": "TERMINATE - Campaign complete. Final document ready for delivery.", "timestamp": time.time() + 15}
    ]
    
    st.session_state.agent_conversation = conversation
    
    # Simulate campaign content
    campaign_content = f"""
# {campaign_params['name']}

## Campaign Overview
A thrilling adventure set in the war-torn kingdom of **Aethelgard**, where ancient powers stir beneath the surface. Perfect for characters from levels {campaign_params['level_range'][0]} to {campaign_params['level_range'][1]}, this campaign blends tactical combat, political intrigue, and deep lore.

## World Setting
The once-great kingdom of Aethelgard is fractured after the assassination of its king. Three factions vie for control:
- **The Iron Covenant**: Noble knights trying to preserve order
- **The Cinder Syndicate**: Merchant lords exploiting the chaos
- **The Veiled Circle**: Cultists seeking to awaken an ancient evil

## Major Story Arcs
1. **The Broken Crown**: Investigate the king's assassination and prevent civil war
2. **Shadows of the Past**: Uncover the cult's plans and stop their ritual at the Moonstone Ruins
3. **Dawn of a New Era**: Rally the factions to face the awakened evil and determine Aethelgard's future

## Key NPCs
- **Ser Aric Thorne**: A disgraced knight seeking redemption (Potential Ally)
- **Lady Elara Voss**: Ruthless merchant lord with hidden agenda (Neutral)
- **Brother Malakar**: Charismatic cult leader with ties to ancient evil (Antagonist)
- **Kaelen Moonshadow**: Elven sage who knows the true history of Aethelgard (Mentor)
- **Grakk Stonefist**: Dwarven blacksmith with secrets about the royal family (Quest Giver)
- **The Whisper**: Mysterious informant who knows too much (Wild Card)

## Notable Encounters
1. **Ambush at Whisper Creek**: Bandits attack the party (Level 1)
2. **The Haunted Manor**: Investigate disappearances in a noble's estate (Level 2)
3. **Arena of Flames**: Forced to fight in an underground arena (Level 3)
4. **Moonstone Ruins**: Dungeon crawl through ancient elven ruins (Level 4)
5. **Cult Ritual**: Stop a summoning ritual at a sacred site (Level 5)

## Final Confrontation
The party must infiltrate the cult's stronghold beneath the capital city. They face Brother Malakar and his lieutenants, then confront the awakened entity. The outcome determines Aethelgard's fate:
- **Restoration**: The kingdom reunites under a new ruler
- **Tyranny**: One faction seizes absolute control
- **New Dawn**: The factions form a council to govern together

## Special Features
- Dynamic faction reputation system
- Player choices significantly alter the story
- Hidden locations with valuable lore and artifacts
- Opportunities for each character class to shine

*This campaign is designed for a party of {campaign_params['party_size']} adventurers. May your dice roll true!*
"""
    return campaign_content

# Sidebar content
with st.sidebar:
    st.image("https://i.imgur.com/8zZ6B0m.png", use_column_width=True)
    st.markdown("<h2 style='color: #D4AF37; text-align: center;'>Campaign Forge</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Ollama connection status
    with st.spinner("Checking Ollama connection..."):
        ollama_models = get_ollama_models()
    
    if ollama_models:
        st.success("✅ Ollama server connected")
        st.session_state.ollama_models = ollama_models
        
        # Model selection dropdown
        selected_model = st.selectbox(
            "Select AI Model",
            ollama_models,
            index=0 if ollama_models else None,
            help="Choose the Ollama model to power your campaign generation"
        )
        st.session_state.selected_model = selected_model
    else:
        st.error("❌ Ollama not detected")
        st.markdown("""
        **To use this tool:**
        1. Install Ollama from [ollama.com](https://ollama.com)
        2. Pull models with `ollama pull mistral` or `ollama pull llama3`
        3. Start the Ollama server
        4. Refresh this page
        """)
        st.session_state.selected_model = None
    
    st.markdown("---")
    st.markdown("### Generation Tips")
    st.markdown("""
    - Be specific with your theme and tone requests
    - Include special elements you want featured
    - Consider your players' preferences
    - Start with lower levels for new groups
    """)
    
    st.markdown("---")
    st.markdown("### System Requirements")
    st.markdown("""
    - Ollama running locally (port 11434)
    - At least 8GB RAM recommended
    - Models: mistral, llama3, or wizard-vicuna
    """)

# Main content
st.markdown("<h1 class='main-header'>🐉 D&D CAMPAIGN FORGE 🧙‍♂️</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #D4AF37; font-size: 1.2rem;'>Craft epic adventures with AI-powered worldbuilding</p>", unsafe_allow_html=True)

# Campaign parameters form
with st.container():
    st.markdown("<h2 class='section-title'>Campaign Specifications</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        campaign_name = st.text_input("Campaign Name*", placeholder="The Shattered Crown", help="A memorable title for your adventure")
        theme = st.text_input("Theme*", placeholder="Political intrigue in a dying empire", help="The central concept or focus")
        tone = st.selectbox("Tone", ["Gritty Realism", "Heroic Fantasy", "Dark Horror", "Lighthearted Adventure", "Mystery & Intrigue", "Epic High Fantasy"])
        
    with col2:
        level_range = st.slider("Character Levels", 1, 20, (1, 5), help="Starting and ending levels for the campaign")
        party_size = st.slider("Party Size", 2, 8, 4, help="Number of player characters")
        world_setting = st.selectbox("World Setting", ["High Fantasy", "Low Magic", "Gothic Horror", "Ancient Ruins", "Underdark", "Planar Travel", "Custom"])
    
    special_requests = st.text_area("Special Requests", placeholder="Include a dragon ally, focus on naval combat, feature a time-travel element, etc.", height=100, help="Anything specific you want included in your campaign")
    
    # Validation and generation button
    if not campaign_name or not theme:
        st.warning("Please provide at least a Campaign Name and Theme to proceed.")
        generate_disabled = True
    else:
        generate_disabled = False if st.session_state.selected_model or st.secrets.get("demo_mode", False) else True
    
    if st.button("✨ Generate Campaign", disabled=generate_disabled, use_container_width=True):
        # Store campaign parameters
        campaign_params = {
            "name": campaign_name,
            "theme": theme,
            "tone": tone,
            "level_range": level_range,
            "party_size": party_size,
            "world_setting": world_setting,
            "special_requests": special_requests
        }
        
        # Reset state
        st.session_state.campaign_generated = False
        st.session_state.campaign_data = None
        st.session_state.agent_conversation = []
        st.session_state.generation_complete = False
        
        # Generate campaign
        if st.session_state.selected_model:
            with st.spinner("Initializing agents..."):
                agents = initialize_agents(st.session_state.selected_model)
            
            st.toast("🧙 Campaign Master has assembled the adventuring party!", icon="🧙‍♂️")
            
            # Run campaign generation in a thread to allow UI updates
            def generation_thread():
                campaign_content = generate_campaign(agents, campaign_params)
                st.session_state.campaign_data = campaign_content
                st.session_state.campaign_generated = True
                
            thread = threading.Thread(target=generation_thread)
            thread.start()
            
            # Show progress while generation is in progress
            progress_placeholder = st.empty()
            thinking_placeholder = st.empty()
            conversation_placeholder = st.empty()
            
            dots = 0
            while thread.is_alive():
                dots = (dots + 1) % 4
                thinking_animation = "." * dots + " " * (3 - dots)
                
                with progress_placeholder:
                    st.markdown(f"<div class='thinking-animation pulse'>{thinking_animation}</div>", unsafe_allow_html=True)
                
                with thinking_placeholder:
                    st.info(f"🧙 Campaign Master is crafting your adventure... (Using {st.session_state.selected_model})")
                
                # Display latest agent messages
                if st.session_state.agent_conversation:
                    with conversation_placeholder:
                        st.markdown("**Agent Collaboration:**")
                        for msg in st.session_state.agent_conversation[-3:]:
                            if msg["sender"] == "UserProxy":
                                st.markdown(f"<div class='user-message'><b>{msg['sender']}</b>: {msg['content']}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='agent-message'><b>{msg['sender']}</b> → {msg['recipient']}: {msg['content']}</div>", unsafe_allow_html=True)
                
                time.sleep(0.5)
            
            # Cleanup after generation
            progress_placeholder.empty()
            thinking_placeholder.empty()
            conversation_placeholder.empty()
            
        else:
            # Demo mode simulation
            st.toast("⚔️ Running in demo mode with simulated generation", icon="⚔️")
            with st.spinner("Generating campaign in demo mode..."):
                time.sleep(3)
                campaign_content = simulate_campaign_generation(campaign_params)
                st.session_state.campaign_data = campaign_content
                st.session_state.campaign_generated = True
                st.session_state.generation_complete = True

# Display generated campaign
if st.session_state.campaign_generated and st.session_state.campaign_data:
    st.markdown("<h2 class='section-title'>.Generated Campaign: " + campaign_name + "</h2>", unsafe_allow_html=True)
    
    # Campaign content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📜 Full Campaign", "🗺️ World Map", "👥 NPCs & Factions", "⚔️ Encounters"])
    
    with tab1:
        st.markdown(st.session_state.campaign_data, unsafe_allow_html=True)
        
        # Download button
        st.download_button(
            label="📥 Download Campaign (Markdown)",
            data=st.session_state.campaign_data,
            file_name=f"{campaign_name.replace(' ', '_')}_campaign.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with tab2:
        st.markdown("### World of Aethelgard")
        st.markdown("""
        **Key Locations:**
        - **Eldoria**: The capital city, now divided among factions
        - **Moonstone Ruins**: Ancient elven site holding dark secrets
        - **The Shattered Peaks**: Mountain range hiding dwarven strongholds
        - **Whisperwood Forest**: Haunted woods with fey creatures
        - **Bloodsalt Mines**: Contested resource area
        """)
        st.markdown("<div class='map-placeholder'>🗺️ Interactive World Map Would Appear Here<br>(Full version would integrate with D&D mapping tools)</div>", unsafe_allow_html=True)
        
        st.markdown("### Factions & Territories")
        st.image("https://i.imgur.com/6QbX6yA.png", caption="Faction Influence Map", use_column_width=True)
    
    with tab3:
        st.markdown("### Major NPCs")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Allies & Neutrals")
            st.markdown("**Ser Aric Thorne** - Disgraced knight seeking redemption\n- *Motivation:* Clear his name and restore honor\n- *Secret:* Knows who really killed the king")
            
            st.markdown("**Kaelen Moonshadow** - Elven sage\n- *Motivation:* Protect ancient knowledge\n- *Connection:* Guardian of the Moonstone Ruins")
        
        with col2:
            st.markdown("#### Antagonists")
            st.markdown("**Brother Malakar** - Cult leader\n- *Motivation:* Awaken the ancient evil to 'purify' the world\n- *Weakness:* Fanatically devoted to his god")
            
            st.markdown("**Lady Elara Voss** - Merchant lord\n- *Motivation:* Control the kingdom's resources\n- *Secret:* Funding both sides of the conflict")
        
        st.markdown("### Faction Relationships")
        st.image("https://i.imgur.com/FLmNwDk.png", caption="Faction Alliance/Conflict Web", use_column_width=True)
    
    with tab4:
        st.markdown("### Key Encounters")
        
        st.markdown("#### Level 1-2")
        st.markdown("- **Ambush at Whisper Creek**: Bandits attack the party (Combat)")
        st.markdown("- **The Haunted Manor**: Investigate disappearances (Exploration/Social)")
        
        st.markdown("#### Level 3-4")
        st.markdown("- **Arena of Flames**: Forced to fight in underground arena (Combat)")
        st.markdown("- **Moonstone Ruins**: Dungeon crawl through ancient ruins (Exploration/Combat)")
        
        st.markdown("#### Level 5")
        st.markdown("- **Cult Ritual**: Stop a summoning ritual (Combat/Social)")
        st.markdown("- **The Final Confrontation**: Battle the awakened entity (Epic Combat)")
        
        st.markdown("### Encounter Balance")
        st.markdown("""
        <div class='stat-block'>
            <div class='stat-item'>
                <div class='stat-value'>8</div>
                <div class='stat-label'>Combat</div>
            </div>
            <div class='stat-item'>
                <div class='stat-value'>5</div>
                <div class='stat-label'>Social</div>
            </div>
            <div class='stat-item'>
                <div class='stat-value'>4</div>
                <div class='stat-label'>Exploration</div>
            </div>
            <div class='stat-item'>
                <div class='stat-value'>3</div>
                <div class='stat-label'>Puzzles</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Agent conversation log (collapsible)
if st.session_state.agent_conversation:
    with st.expander("🔍 View Agent Collaboration Log"):
        for msg in st.session_state.agent_conversation:
            timestamp = time.strftime('%H:%M:%S', time.localtime(msg["timestamp"]))
            if msg["sender"] == "UserProxy":
                st.markdown(f"<div class='user-message'><b>[{timestamp}] {msg['sender']}</b>: {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='agent-message'><b>[{timestamp}] {msg['sender']}</b> → {msg['recipient']}: {msg['content']}</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class='footer'>
    <p>D&D Campaign Forge | Powered by AutoGen & Ollama | For entertainment purposes only | Not affiliated with Wizards of the Coast</p>
</div>
""", unsafe_allow_html=True)

# Show connection warning if needed
if not ollama_models and not st.secrets.get("demo_mode", False):
    st.warning("""
    **Ollama Server Not Detected**
    
    This tool requires Ollama to be running locally. To generate real campaigns:
    
    1. Download and install Ollama from [ollama.com](https://ollama.com)
    2. Pull a model: `ollama pull mistral` or `ollama pull llama3`
    3. Start the Ollama server
    4. Refresh this page
    
    Until then, you can explore the demo mode by entering campaign details and clicking "Generate Campaign".
    """)