import dash_bootstrap_components as dbc

coffee_color = "#4B3832"
text_color = "#F8F1E7"

nav_items = [
    ("🏠 ホーム", "/"),
    ("🧪 次の実験", "/experiment"),
    ("🛠️ 評価・履歴", "/evaluation"),
    ("📊 分析", "/analysis"),
    ("⚙️ 設定", "/settings"),
]

nav_links = [
    dbc.NavLink(label, href=href, active="exact",
                id={"type": "nav-link", "index": i})
    for i, (label, href) in enumerate(nav_items)
]

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
                nav_links,
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
    sticky="top",
    className="w-100 p-0 m-0",
    style={"margin": "0", "padding": "0"},
)
