# ---------------------------------------------------------------------------
# SA Reference Data & Shared Logic
# ---------------------------------------------------------------------------

SA_PROVINCES = [
    "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
    "Limpopo", "Mpumalanga", "North West", "Free State", "Northern Cape",
]

SUBURBS_BY_PROVINCE = {
    "Gauteng": ["Sandton", "Rosebank", "Morningside", "Fourways", "Midrand", "Centurion", "Pretoria East", "Bryanston", "Randburg", "Edenvale", "Bedfordview", "Boksburg", "Soweto", "Alexandra", "Maboneng", "Northcliff", "Westcliff", "Houghton", "Parktown", "Melville"],
    "Western Cape": ["Camps Bay", "Clifton", "Sea Point", "Green Point", "Waterfront", "Constantia", "Bishopscourt", "Claremont", "Rondebosch", "Newlands", "Stellenbosch", "Paarl", "Somerset West", "Strand", "George", "Knysna", "Plettenberg Bay", "Franschhoek", "Hermanus", "Moss Bay"],
    "KwaZulu-Natal": ["Umhlanga", "Ballito", "La Lucia", "Durban North", "Berea", "Glenwood", "Westville", "Hillcrest", "Pinetown", "Amanzimtoti", "Pietermaritzburg", "Howick", "Margate", "Port Shepstone", "Richards Bay"],
    "Eastern Cape": ["Gqeberha (Port Elizabeth)", "Summerstrand", "Humewood", "Jeffreys Bay", "East London", "Vincent", "Beacon Bay", "Mdantsane", "Grahamstown"],
    "Limpopo": ["Polokwane", "Tzaneen", "Phalaborwa", "Louis Trichardt", "Mokopane"],
    "Mpumalanga": ["Mbombela (Nelspruit)", "White River", "Hazyview", "Witbank (eMalahleni)", "Secunda"],
    "North West": ["Rustenburg", "Potchefstroom", "Klerksdorp", "Hartbeespoort", "Brits"],
    "Free State": ["Bloemfontein", "Welkom", "Bethlehem", "Sasolburg", "Parys"],
    "Northern Cape": ["Kimberley", "Upington", "Springbok", "De Aar", "Kuruman"],
}

PROPERTY_TYPES = ["House", "Apartment", "Townhouse", "Cluster", "Vacant Land", "Farm", "Commercial"]

def calculate_transfer_duty(price: float) -> float:
    """Calculate SARS transfer duty for 2026/27 tax year."""
    if price <= 1_210_000: return 0.0
    elif price <= 1_663_800: return (price - 1_210_000) * 0.03
    elif price <= 2_329_300: return 13_614 + (price - 1_663_800) * 0.06
    elif price <= 2_994_800: return 53_544 + (price - 2_329_300) * 0.08
    elif price <= 13_310_000: return 106_784 + (price - 2_994_800) * 0.11
    else: return 1_241_456 + (price - 13_310_000) * 0.13