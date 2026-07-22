"""Mapa equipo -> confederación, con la grafía EXACTA del dataset martj42.

Cubre todas las selecciones FIFA + entidades históricas (Czechoslovakia, Yugoslavia,
German DR, Vietnam Republic, etc.). Las asociaciones que cambiaron de confederación
(Israel, Kazakhstan -> UEFA; Australia -> AFC) se asignan a su confederación ACTUAL.
Todo lo no listado (equipos regionales/no-FIFA: Basque Country, Sápmi, Shetland...)
cae en 'Other' vía .get(team, 'Other').
"""

CONFEDERATION = {}


def _add(conf, teams):
    for t in teams:
        CONFEDERATION[t] = conf


_add("UEFA", [
    "Sweden", "England", "Germany", "Hungary", "France", "Italy", "Poland",
    "Switzerland", "Netherlands", "Norway", "Denmark", "Austria", "Belgium",
    "Scotland", "Finland", "Spain", "Romania", "Russia", "Bulgaria", "Wales",
    "Northern Ireland", "Portugal", "Turkey", "Republic of Ireland", "Greece",
    "Estonia", "Iceland", "Czechoslovakia", "Yugoslavia", "Israel", "Croatia",
    "Albania", "Slovakia", "Czech Republic", "Serbia", "Ukraine", "Slovenia",
    "Belarus", "Azerbaijan", "Georgia", "North Macedonia", "Moldova",
    "Faroe Islands", "Bosnia and Herzegovina", "Armenia", "Kazakhstan",
    "Liechtenstein", "Latvia", "Lithuania", "Cyprus", "Luxembourg", "Malta",
    "German DR", "Montenegro", "Gibraltar", "Andorra", "San Marino", "Kosovo",
])

_add("CONMEBOL", [
    "Argentina", "Brazil", "Uruguay", "Chile", "Paraguay", "Peru", "Colombia",
    "Ecuador", "Bolivia", "Venezuela",
])

_add("CONCACAF", [
    "Mexico", "United States", "Trinidad and Tobago", "Costa Rica", "Jamaica",
    "Honduras", "El Salvador", "Guatemala", "Panama", "Haiti", "Canada", "Cuba",
    "Suriname", "Curaçao", "Martinique", "Guadeloupe", "Guyana", "Barbados",
    "Grenada", "Saint Vincent and the Grenadines", "French Guiana",
    "Antigua and Barbuda", "Saint Lucia", "Saint Kitts and Nevis", "Nicaragua",
    "Dominica", "Bermuda", "Puerto Rico", "Dominican Republic", "Belize",
    "Aruba", "Cayman Islands", "Bahamas", "British Virgin Islands",
    "United States Virgin Islands", "Turks and Caicos Islands", "Saint Martin",
    "Montserrat", "Anguilla", "Sint Maarten", "Bonaire", "Saint Barthélemy",
])

_add("AFC", [
    "South Korea", "Thailand", "Malaysia", "Japan", "Saudi Arabia", "Indonesia",
    "Singapore", "China PR", "Kuwait", "Iraq", "Qatar", "Iran",
    "United Arab Emirates", "Bahrain", "Oman", "India", "Myanmar", "Jordan",
    "Syria", "Hong Kong", "North Korea", "Uzbekistan", "Philippines", "Lebanon",
    "Vietnam", "Cambodia", "Nepal", "Yemen", "Sri Lanka", "Pakistan",
    "Palestine", "Taiwan", "Maldives", "Laos", "Bangladesh", "Tajikistan",
    "Vietnam Republic", "Kyrgyzstan", "Turkmenistan", "Macau", "Afghanistan",
    "Brunei", "Guam", "Bhutan", "Timor-Leste", "Mongolia", "Yemen DPR",
    "North Vietnam", "Northern Mariana Islands", "Australia",
])

_add("CAF", [
    "Zambia", "Egypt", "Kenya", "Uganda", "Tunisia", "Ghana", "Nigeria",
    "Malawi", "Senegal", "Ivory Coast", "Morocco", "Algeria", "Cameroon",
    "Tanzania", "Mali", "Guinea", "DR Congo", "Zimbabwe", "Sudan",
    "Burkina Faso", "South Africa", "Togo", "Angola", "Ethiopia", "Congo",
    "Gabon", "Libya", "Mozambique", "Benin", "Madagascar", "Botswana",
    "Mauritius", "Liberia", "Sierra Leone", "Lesotho", "Mauritania", "Gambia",
    "Rwanda", "Namibia", "Eswatini", "Niger", "Cape Verde", "Burundi",
    "Zanzibar", "Guinea-Bissau", "Equatorial Guinea", "Seychelles", "Comoros",
    "Chad", "Central African Republic", "Réunion", "Somalia", "Djibouti",
    "South Sudan", "São Tomé and Príncipe", "Eritrea", "Mayotte",
])

_add("OFC", [
    "New Zealand", "Fiji", "New Caledonia", "Tahiti", "Solomon Islands",
    "Vanuatu", "Papua New Guinea", "Samoa", "Tonga", "American Samoa",
    "Cook Islands", "Tuvalu", "Kiribati", "Niue", "Micronesia",
    "Wallis Islands and Futuna",
])


def confederation(team: str) -> str:
    """Confederación del equipo, o 'Other' si no es selección FIFA reconocida."""
    return CONFEDERATION.get(team, "Other")
