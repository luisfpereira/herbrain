"""Simple GPT chat component for the AI prediction page."""

import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, callback
import openai
from polpo.dash.style import STYLE as S

def gpt_chat_component():
    """Create a simple GPT chat component with a text input that shows responses in the same box."""
    
    return html.Div([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H4("Ask ChatGPT about the data", style={"fontSize": S.title_fontsize}),
                html.P(
                    "Ask questions about the brain changes during pregnancy, hormone levels, or any other aspects of the data shown on this page.",
                    style={"fontSize": S.text_fontsize, "fontFamily": S.text_fontfamily}
                ),
                dbc.Textarea(
                    id="gpt-input",
                    placeholder="Type your question here...",
                    style={"width": "100%", "height": "100px", "marginBottom": "10px"}
                ),
                dbc.Button("Ask ChatGPT", id="gpt-submit", color="primary", className="mb-3"),
            ])
        ])
    ])

@callback(
    Output("gpt-input", "value"),
    Input("gpt-submit", "n_clicks"),
    State("gpt-input", "value"),
    State("gestWeek-slider", "value"),  # Gestational week slider
    State("estro-slider", "value"),     # Estrogen slider
    State("prog-slider", "value"),      # Progesterone slider
    State("lh-slider", "value"),        # LH slider
    prevent_initial_call=True
)
def update_gpt_response(n_clicks, question, gest_week, estro, prog, lh):
    """Update the textarea with GPT's response when the submit button is clicked."""
    if not question:
        return "Please enter a question."
    
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
                {"role": "system", "content": "You are a helpful assistant explaining brain changes during pregnancy. Focus on the relationship between hormones and brain structure."},
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}" 