"""Simple GPT chat component for the AI prediction page."""

import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, callback, dcc, ctx
import openai
from polpo.dash.style import STYLE as S

def gpt_chat_component():
    """Create a GPT chat component with message history and chat-like interface."""
    
    return html.Div([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H4("Ask ChatGPT about the data", style={"fontSize": S.title_fontsize}),
                html.P(
                    "Ask questions about the brain changes during pregnancy, hormone levels, or any other aspects of the data shown on this page.",
                    style={"fontSize": S.text_fontsize, "fontFamily": S.text_fontfamily}
                ),
                # Chat history container
                html.Div(
                    id="chat-history",
                    style={
                        "height": "400px",
                        "overflowY": "auto",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                        "padding": "10px",
                        "marginBottom": "10px",
                        "backgroundColor": "#f8f9fa"
                    }
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
     State("lh-slider", "value")],       # LH slider
    prevent_initial_call=True
)
def update_chat(n_clicks, question, chat_history, gest_week, estro, prog, lh):
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

Please use these values to provide context in your response."""
        
        # Create the chat completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a faster model
            messages=[
                {"role": "system", "content": "You are a neuroscientist specializing in the pregnancy and postpartum brain. You answer questions using short, precise sentences. Only respond to questions related to neuroscience of pregnancy, hormones, and motherhood, and women's brains. If a question is outside this scope, politely decline to answer."},
                {"role": "system", "content": "You are a helpful assistant explaining brain changes during pregnancy. Focus on the relationship between hormones and brain structure."},
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ]
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