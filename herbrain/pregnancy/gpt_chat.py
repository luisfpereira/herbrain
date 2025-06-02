"""Simple GPT chat component for the AI prediction page."""

import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, callback, dcc, ctx
import openai
from polpo.dash.style import STYLE as S
import base64
import plotly.io as pio
import plotly.graph_objects as go
import io

def gpt_chat_component():
    """Create a GPT chat component with message history and chat-like interface."""
    
    return html.Div([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H4("Ask the AI neurobot what the changes mean for your brain", style={"fontSize": S.title_fontsize}),
                html.P(
                    "Ask questions about the brain changes during pregnancy, hormone levels, or any other aspects of the data shown on this page. The AI neurobot will analyze the 3D visualization of your brain and provide insights.",
                    style={"fontSize": "0.9em", "fontFamily": S.text_fontfamily, "color": "#666"}
                ),
                # Input area
                dbc.Row([
                    dbc.Col([
                        dbc.Textarea(
                            id="gpt-input",
                            placeholder="Type your question here...",
                            style={"width": "100%", "height": "100px", "marginBottom": "10px"}
                        ),
                        dbc.Button("Ask ChatGPT", id="gpt-submit", color="primary", className="mb-3"),
                    ])
                ]),
                # Chat history container
                html.Div(
                    id="chat-history",
                    style={
                        "height": "200px",  # Reduced from 400px to 200px
                        "overflowY": "auto",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                        "padding": "10px",
                        "marginBottom": "10px",
                        "backgroundColor": "#f8f9fa"
                    }
                ),
                # Store for chat history
                dcc.Store(id="chat-store", data=[]),
            ])
        ])
    ])

def create_message_bubble(message, is_user=True):
    """Create a message bubble for the chat."""
    return html.Div(
        [
            html.Div(
                message,
                style={
                    "backgroundColor": "#007bff" if is_user else "#e9ecef",
                    "color": "white" if is_user else "black",
                    "padding": "10px",
                    "borderRadius": "10px",
                    "marginBottom": "10px",
                    "maxWidth": "80%",
                    "marginLeft": "auto" if is_user else "0",
                    "marginRight": "0" if is_user else "auto",
                }
            )
        ],
        style={"marginBottom": "10px"}
    )

@callback(
    [Output("chat-history", "children"),
     Output("gpt-input", "value")],
    Input("gpt-submit", "n_clicks"),
    [State("gpt-input", "value"),
     State("chat-store", "data"),
     State("gestWeek-slider", "value"),  # Gestational week slider
     State("estro-slider", "value"),     # Estrogen slider
     State("prog-slider", "value"),      # Progesterone slider
     State("lh-slider", "value"),        # LH slider
     State("mesh-plot", "figure")],      # Current mesh figure
    prevent_initial_call=True
)
def update_chat(n_clicks, question, chat_history, gest_week, estro, prog, lh, figure):
    """Update the chat history when a new message is sent."""
    if not question:
        return chat_history, ""
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI()
        
        # Create context string with current slider values
        context = f"""Current hormone levels and gestational week:
- Gestational Week: {gest_week}
- Estrogen: {estro} pg/ml
- Progesterone: {prog} ng/ml
- LH: {lh} ng/ml

Please analyze the attached 3D mesh visualization and use these hormone values to provide context in your response."""
        
        # Convert Plotly figure to image
        if figure:
            # Create a proper Plotly figure object from the dictionary
            temp_fig = go.Figure(figure)
            # Ensure the figure has the right size and layout
            temp_fig.update_layout(
                width=800,
                height=600,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            # Convert to PNG image
            img_bytes = pio.to_image(temp_fig, format="png")
            # Convert to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        else:
            img_base64 = None
        
        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": "You are a neuroscientist specializing in the pregnancy and postpartum brain. You answer questions using short, precise sentences. Only respond to questions related to neuroscience of pregnancy, hormones, and motherhood, and women's brains. If a question is outside this scope, politely decline to answer."},
            {"role": "system", "content": "You are a helpful assistant explaining brain changes during pregnancy. Focus on the relationship between hormones and brain structure."},
            {"role": "system", "content": "Just above your chat box, you see the rendered 3D hippocampus of a brain of a pregnant woman—this is the image provided in your context. Be prepared to answer questions based on what you observe in this brain image."},
            {"role": "system", "content": context}
        ]
        
        # Add the image if available
        if img_base64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": question})
        
        # Create the chat completion
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o-mini to handle mesh visualization
            messages=messages,
            max_tokens=500
        )
        
        # Get the response
        answer = response.choices[0].message.content
        
        # Update chat history
        new_history = chat_history + [
            create_message_bubble(question, is_user=True),
            create_message_bubble(answer, is_user=False)
        ]
        
        return new_history, ""
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        new_history = chat_history + [
            create_message_bubble(question, is_user=True),
            create_message_bubble(error_message, is_user=False)
        ]
        return new_history, ""

@callback(
    Output("gpt-submit", "n_clicks"),
    Input("gpt-input", "n_submit"),
    State("gpt-submit", "n_clicks"),
    prevent_initial_call=True
)
def handle_enter(n_submit, n_clicks):
    """Handle Enter key press in the textarea."""
    if n_submit is None:
        return n_clicks
    return (n_clicks or 0) + 1 