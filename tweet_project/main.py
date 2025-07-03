from src import preprocessing, modeling
import os

def main():
    data_path = os.path.join('./data', 'tweets.csv')
    
    # Check scikit-learn est disponible
    if not modeling.SKLEARN_AVAILABLE:
        print("⚠️  scikit-learn n'est pas installé.")
        print("Pour utiliser les fonctionnalités de modélisation, installez scikit-learn avec :")
        print("pip install scikit-learn")
        print()
        
        print("📊 Démonstration du chargement des données :")
        try:
            texts, targets = modeling.load_texts_targets(data_path)
            print(f"✅ Données chargées avec succès !")
            print(f"   - Nombre d'échantillons : {len(texts)}")
            print(f"   - Classes cibles : {set(targets)}")
            print(f"   - Exemple de texte : '{texts[0][:100]}...'")
            
            print("\n🔧 Démonstration du préprocessing :")
            clean_texts = preprocessing.build_clean_corpus(texts[:5], stem_words=False)
            print(f"   - Texte original : '{texts[0]}'")
            print(f"   - Texte nettoyé : '{clean_texts[0]}'")
            
        except FileNotFoundError:
            print(f"❌ Fichier de données non trouvé : {data_path}")
            print("   Assurez-vous que le fichier tweets.csv existe dans le dossier data/")
        except Exception as e:
            print(f"❌ Erreur lors du chargement : {e}")
    else:
        print("🤖 Entraînement du modèle en cours...")
        try:
            metrics = modeling.train_evaluate(data_path)
            print("✅ Entraînement terminé !")
            print("📈 Métriques du modèle :")
            for metric, value in metrics.items():
                print(f"   - {metric.capitalize()}: {value:.4f}")
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement : {e}")

if __name__ == '__main__':
    main()
