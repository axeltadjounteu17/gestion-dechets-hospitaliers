import pandas as pd
import random

# Load dataset
file_path = '/home/axel-renaud/Images/PROJET_INF_365/notebooks/data/dechets_hospitaliers.csv'
df = pd.read_csv(file_path)

# Dictionary of real data
real_data = {
    "Sénégal": {
        "regions": ["Dakar", "Thiès", "Diourbel", "Saint-Louis", "Ziguinchor", "Kaolack", "Fatick", "Kolda"],
        "hospitals": [
            "Hôpital Principal de Dakar", "Hôpital Aristide Le Dantec", "CHNU de Fann", 
            "Hôpital Général Idrissa Pouye", "Hôpital Roi Baudouin", "Hôpital Régional de Thiès",
            "Hôpital Régional de Saint-Louis", "Hôpital de la Paix de Ziguinchor", "Centre de Santé de Mbour",
            "Hôpital Régional de Kaolack", "Hôpital Ndamatou", "Clinique de la Madeleine", "Clinique Casamance"
        ]
    },
    "Cameroun": {
        "regions": ["Centre", "Littoral", "Ouest", "Nord", "Adamaoua", "Est", "Sud", "Nord-Ouest"],
        "hospitals": [
            "Hôpital Central de Yaoundé", "Hôpital Général de Yaoundé", "CHU de Yaoundé",
            "Hôpital Laquintinie de Douala", "Hôpital Général de Douala", "Hôpital Gynéco-Obstétrique",
            "Hôpital Régional de Bafoussam", "Hôpital Régional de Garoua", "Hôpital Régional de Ngaoundéré",
            "Hôpital Régional de Bertoua", "Hôpital Régional d'Ebolowa", "Hôpital Régional de Bamenda",
            "Clinique Bonanjo", "Polyclinique Sainte-Anne"
        ]
    },
    "Togo": {
        "regions": ["Maritime", "Plateaux", "Centrale", "Kara", "Savanes"],
        "hospitals": [
            "CHU Sylvanus Olympio", "CHU Campus", "CHR Lomé Commune", "Hôpital de Bè",
            "CHR Tsévié", "CHR Atakpamé", "CHR Sokodé", "CHU Kara", "CHR Dapaong",
            "Hôpital Dogta-Lafiè", "Clinique Biasa", "Clinique Autel d'Elie"
        ]
    },
    "Gabon": {
        "regions": ["Estuaire", "Haut-Ogooué", "Ogooué-Maritime", "Ngounié", "Woleu-Ntem"],
        "hospitals": [
            "CHU de Libreville", "Hôpital d'Instruction des Armées Omar Bongo", 
            "Hôpital d'Instruction des Armées Akanda", "Fondation Jeanne Ebori", "Hôpital Albert Schweitzer",
            "CHR de Franceville", "CHR de Port-Gentil", "Hôpital Régional de Mouila", "CHR d'Oyem",
            "Polyclinique El Rapha", "Clinique Chambrier"
        ]
    },
    "Bénin": {
        "regions": ["Littoral", "Atlantique", "Ouémé", "Borgou", "Zou", "Atacora"],
        "hospitals": [
            "CNHU Hubert Koutoukou Maga", "Hôpital de Zone de Ménontin", "Hôpital de la Mère et de l'Enfant",
            "CHU de Suru-Léré", "CHUD Ouémé-Plateau", "CHUD Borgou", "CHD Zou", "CHD Atacora",
            "Hôpital Saint-Luc", "Clinique Boni", "Clinique Mahouna"
        ]
    },
    "RDC": {
        "regions": ["Kinshasa", "Haut-Katanga", "Nord-Kivu", "Kongo Central", "Lualaba", "Sud-Kivu"],
        "hospitals": [
            "Hôpital Général de Référence de Kinshasa", "Clinique Ngaliema", "Hôpital du Cinquantenaire",
            "Centre Médical de Kinshasa", "Hôpital Sendwe (Lubumbashi)", "Hôpital Général de Goma",
            "Hôpital Heal Africa", "Hôpital de Panzi", "Hôpital Général de Matadi",
            "Clinique Universitaire de Kinshasa"
        ]
    },
    "Côte d'Ivoire": {
        "regions": ["Abidjan", "Lagunes", "Haut-Sassandra", "Gbêkê", "San-Pédro", "Poro"],
        "hospitals": [
            "CHU de Cocody", "CHU de Treichville", "CHU de Yopougon", "Hôpital Mère-Enfant de Bingerville",
            "Hôpital Militaire d'Abidjan", "CHR de San-Pédro", "CHU de Bouaké", "CHR de Daloa",
            "CHR de Korhogo", "Polyclinique Internationale Sainte Anne-Marie (PISAM)", "Clinique Farah"
        ]
    }
}

# Update the dataframe
def update_row(row):
    country = row['pays']
    if country in real_data:
        # Assign a random region from the country
        new_region = random.choice(real_data[country]['regions'])
        # Assign a random hospital from the country
        new_hospital = random.choice(real_data[country]['hospitals'])
        return new_region, new_hospital
    else:
        return row['region'], row['hopital']

# Apply the update
df[['region', 'hopital']] = df.apply(lambda row: pd.Series(update_row(row)), axis=1)

# Save the updated file
df.to_csv(file_path, index=False)
print(f"File updated successfully: {file_path}")

# Also update the one in web_app just in case, though the notebook writes there usually. 
# But let's keep the source of truth updated.
web_app_data_path = '/home/axel-renaud/Images/PROJET_INF_365/web_app/dechets_hospitaliers.csv'
df.to_csv(web_app_data_path, index=False)
print(f"Web app copy updated: {web_app_data_path}")
