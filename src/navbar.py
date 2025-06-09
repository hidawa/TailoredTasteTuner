import dash_bootstrap_components as dbc

coffee_color = "#4B3832"
text_color = "#F8F1E7"

navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row(
            [
                dbc.Col(
                    dbc.NavbarToggler(id="navbar-toggler"),
                    width="auto",
                ),
                dbc.Col(
                    dbc.NavbarBrand(
                        "🍽️ Tailored Taste Tuner",
                        href="/",
                        style={
                            "fontSize": "1.5rem",
                            "fontWeight": "bold",
                            "color": text_color,
                            "whiteSpace": "nowrap"
                        }
                    ),
                    width="auto",
                ),
            ],
            align="center",
            className="g-0",
            style={"flexWrap": "nowrap"},
        ),
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavLink("🏠 ホーム", href="/", active="exact"),
                    dbc.NavLink("🧪 次の実験", href="/experiment", active="exact"),
                    dbc.NavLink("🛠️ 評価・履歴", href="/evaluation",
                                active="exact"),
                    dbc.NavLink("📊 分析", href="/analysis", active="exact"),
                    dbc.NavLink("⚙️ 設定", href="/settings", active="exact"),
                ],
                className="ms-auto",
                navbar=True,
            ),
            id="navbar-collapse",
            navbar=True,
        ),
    ],
        fluid=True,
    ),
    color=coffee_color,
    dark=True,
    expand="lg",
    className="w-100 p-0 m-0",
    style={"margin": "0", "padding": "0"},
)
