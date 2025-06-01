import dash_bootstrap_components as dbc

# カラーテーマ
coffee_color = "#4B3832"
text_color = "#F8F1E7"

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                "🍽️ Tailored Taste Tuner ☕",
                style={
                    "fontSize": "2rem",
                    "fontWeight": "bold",
                    "color": text_color,
                    # "letterSpacing": "0.05em",
                    # "marginRight": "1rem",
                }
            ),
        ],
    ),
    color=coffee_color,
    dark=True,
)
