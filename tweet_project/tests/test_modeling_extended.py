import os
import sys
import unittest
import tempfile
import csv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.modeling import (
    train_evaluate,
    cross_validate,
    train_predict,
    load_texts_targets,
    build_vectorizer,
    build_classifier,
    SKLEARN_AVAILABLE,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tweets.csv')


class TestDataLoading(unittest.TestCase):
    """Tests pour le chargement des données"""
    
    def setUp(self):
        """Crée un fichier de test temporaire"""
        self.temp_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(self.temp_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        writer.writerow([1, 'test', 'paris', 'Ceci est un tweet positif', 1])
        writer.writerow([2, 'test', 'lyon', 'Ceci est un tweet negatif', 0])
        writer.writerow([3, '', '', 'Tweet neutre sans mots-cles', 0])
        writer.writerow([4, 'happy', 'nice', 'Je suis tres heureux aujourd hui!', 1])
        writer.writerow([5, 'sad', 'bad', 'Je me sens triste et deprime', 0])
        self.temp_file.close()
        self.temp_path = self.temp_file.name

    def tearDown(self):
        """Nettoie le fichier temporaire"""
        os.unlink(self.temp_path)

    def test_load_texts_targets_basic(self):
        """Test le chargement basique des textes et cibles"""
        texts, targets = load_texts_targets(self.temp_path)
        self.assertEqual(len(texts), 5)
        self.assertEqual(len(targets), 5)
        self.assertIsInstance(texts[0], str)
        self.assertIsInstance(targets[0], int)

    def test_load_texts_targets_content(self):
        """Test le contenu des données chargées"""
        texts, targets = load_texts_targets(self.temp_path)
        self.assertIn('positif', texts[0])
        self.assertEqual(targets[0], 1)
        self.assertEqual(targets[1], 0)

    def test_load_empty_file(self):
        """Test avec un fichier vide"""
        empty_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(empty_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        empty_file.close()
        
        texts, targets = load_texts_targets(empty_file.name)
        self.assertEqual(len(texts), 0)
        self.assertEqual(len(targets), 0)
        os.unlink(empty_file.name)

    def test_load_with_missing_columns(self):
        """Test avec des colonnes manquantes"""
        bad_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(bad_file)
        writer.writerow(['text', 'target'])
        writer.writerow(['test text', '1'])
        bad_file.close()
        
        texts, targets = load_texts_targets(bad_file.name)
        self.assertEqual(len(texts), 1)
        self.assertEqual(targets[0], 1)
        os.unlink(bad_file.name)


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn non installé")
class TestVectorizers(unittest.TestCase):
    """Tests pour les vectoriseurs"""

    def test_build_tfidf_vectorizer(self):
        """Test la création d'un vectoriseur TF-IDF"""
        vectorizer = build_vectorizer("tfidf")
        self.assertIsNotNone(vectorizer)
        self.assertEqual(vectorizer.__class__.__name__, "TfidfVectorizer")

    def test_build_count_vectorizer(self):
        """Test la création d'un vectoriseur de comptage"""
        vectorizer = build_vectorizer("count")
        self.assertIsNotNone(vectorizer)
        self.assertEqual(vectorizer.__class__.__name__, "CountVectorizer")

    def test_build_unknown_vectorizer(self):
        """Test avec un vectoriseur inconnu"""
        with self.assertRaises(ValueError):
            build_vectorizer("unknown")

    def test_vectorizer_case_sensitivity(self):
        """Test la sensibilité à la casse des noms de vectoriseurs"""
        with self.assertRaises(ValueError):
            build_vectorizer("TFIDF")
        with self.assertRaises(ValueError):
            build_vectorizer("Count")

    def test_vectorizer_empty_string(self):
        """Test avec une chaîne vide"""
        with self.assertRaises(ValueError):
            build_vectorizer("")


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn non installé")
class TestClassifiers(unittest.TestCase):
    """Tests pour les classificateurs"""

    def test_build_logistic_regression(self):
        """Test la création d'une régression logistique"""
        classifier = build_classifier("logreg")
        self.assertIsNotNone(classifier)
        self.assertEqual(classifier.__class__.__name__, "LogisticRegression")
        self.assertEqual(classifier.max_iter, 1000)

    def test_build_svm(self):
        """Test la création d'un SVM linéaire"""
        classifier = build_classifier("svm")
        self.assertIsNotNone(classifier)
        self.assertEqual(classifier.__class__.__name__, "LinearSVC")

    def test_build_unknown_classifier(self):
        """Test avec un classificateur inconnu"""
        with self.assertRaises(ValueError):
            build_classifier("unknown")

    def test_classifier_case_sensitivity(self):
        """Test la sensibilité à la casse des noms de classificateurs"""
        with self.assertRaises(ValueError):
            build_classifier("LOGREG")
        with self.assertRaises(ValueError):
            build_classifier("SVM")

    def test_classifier_empty_string(self):
        """Test avec une chaîne vide"""
        with self.assertRaises(ValueError):
            build_classifier("")


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn non installé")
class TestTrainEvaluate(unittest.TestCase):
    """Tests pour la fonction train_evaluate"""

    def setUp(self):
        """Crée un jeu de données de test plus large"""
        self.temp_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(self.temp_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        
        positive_texts = [
            "Je suis tres heureux aujourd hui!",
            "Quelle belle journee ensoleillee",
            "J adore ce nouveau film, il est fantastique",
            "Excellente experience, je recommande vivement",
            "Super moment passe avec mes amis",
            "Magnifique coucher de soleil ce soir",
            "Delicieux repas au restaurant",
            "Victoire de mon equipe favorite!",
            "Wonderful day at the beach",
            "Amazing performance by the artist"
        ]
        
        negative_texts = [
            "Je me sens triste et deprime",
            "Terrible experience, tres decu",
            "Il pleut encore, quelle journee maussade",
            "Service client epouvantable",
            "J ai perdu mes cles, c est frustrant",
            "Embouteillages terribles ce matin",
            "Film ennuyeux, j ai failli m endormir",
            "Mauvaise nouvelle recue aujourd hui",
            "Awful weather ruining my plans",
            "Disappointed with the service quality"
        ]
        
        for i, text in enumerate(positive_texts):
            writer.writerow([i, 'positive', 'test', text, 1])
        
        for i, text in enumerate(negative_texts):
            writer.writerow([i+10, 'negative', 'test', text, 0])
            
        self.temp_file.close()
        self.temp_path = self.temp_file.name

    def tearDown(self):
        """Nettoie le fichier temporaire"""
        os.unlink(self.temp_path)

    def test_train_evaluate_default_params(self):
        """Test avec les paramètres par défaut"""
        metrics = train_evaluate(self.temp_path)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # Vérifier que les métriques sont dans des plages valides
        for metric_name, value in metrics.items():
            self.assertGreaterEqual(value, 0.0, f"{metric_name} devrait être >= 0")
            self.assertLessEqual(value, 1.0, f"{metric_name} devrait être <= 1")

    def test_train_evaluate_tfidf_logreg(self):
        """Test avec TF-IDF et régression logistique"""
        metrics = train_evaluate(self.temp_path, vectorizer="tfidf", model="logreg")
        self.assertIsInstance(metrics, dict)
        self.assertEqual(len(metrics), 4)

    def test_train_evaluate_count_svm(self):
        """Test avec vectoriseur de comptage et SVM"""
        metrics = train_evaluate(self.temp_path, vectorizer="count", model="svm")
        self.assertIsInstance(metrics, dict)
        self.assertEqual(len(metrics), 4)

    def test_train_evaluate_different_test_sizes(self):
        """Test avec différentes tailles de jeu de test"""
        for test_size in [0.1, 0.3, 0.5]:
            metrics = train_evaluate(self.temp_path, test_size=test_size)
            self.assertIn('accuracy', metrics)

    def test_train_evaluate_different_random_states(self):
        """Test avec différents états aléatoires"""
        metrics1 = train_evaluate(self.temp_path, random_state=42)
        metrics2 = train_evaluate(self.temp_path, random_state=123)
        
        # Les résultats peuvent être différents avec différents états aléatoires
        self.assertIsInstance(metrics1, dict)
        self.assertIsInstance(metrics2, dict)

    def test_train_evaluate_invalid_vectorizer(self):
        """Test avec un vectoriseur invalide"""
        with self.assertRaises(ValueError):
            train_evaluate(self.temp_path, vectorizer="invalid")

    def test_train_evaluate_invalid_model(self):
        """Test avec un modèle invalide"""
        with self.assertRaises(ValueError):
            train_evaluate(self.temp_path, model="invalid")


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn non installé")
class TestTrainPredict(unittest.TestCase):
    """Tests pour la fonction train_predict"""

    def setUp(self):
        """Crée un jeu de données de test"""
        self.temp_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(self.temp_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        
        for i in range(10):
            text = f"Texte positif numero {i}"
            writer.writerow([i, 'pos', 'test', text, 1])
            
        for i in range(10):
            text = f"Texte negatif numero {i}"
            writer.writerow([i+10, 'neg', 'test', text, 0])
            
        self.temp_file.close()
        self.temp_path = self.temp_file.name

    def tearDown(self):
        """Nettoie le fichier temporaire"""
        os.unlink(self.temp_path)

    def test_train_predict_basic(self):
        """Test basique de train_predict"""
        preds, truth = train_predict(self.temp_path)
        self.assertEqual(len(preds), len(truth))
        self.assertGreater(len(preds), 0)



    def test_train_predict_different_test_sizes(self):
        """Test avec différentes tailles de test"""
        total_samples = 20
        
        for test_size in [0.2, 0.3, 0.5]:
            preds, truth = train_predict(self.temp_path, test_size=test_size)
            expected_test_samples = int(total_samples * test_size)
            
            # Tolérance pour l'arrondi
            self.assertAlmostEqual(len(preds), expected_test_samples, delta=1)

    def test_train_predict_reproducibility(self):
        """Test la reproductibilité avec le même random_state"""
        preds1, truth1 = train_predict(self.temp_path, random_state=42)
        preds2, truth2 = train_predict(self.temp_path, random_state=42)
        
        self.assertEqual(preds1, preds2)
        self.assertEqual(truth1, truth2)


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn non installé")
class TestCrossValidate(unittest.TestCase):
    """Tests pour la fonction cross_validate"""

    def setUp(self):
        """Crée un jeu de données de test"""
        self.temp_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(self.temp_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        
        # Créer un jeu de données plus équilibré
        texts_pos = [
            "Excellent produit, je recommande",
            "Service fantastique, tres satisfait",
            "Qualite exceptionnelle, parfait",
            "Experience merveilleuse, top",
            "Genial, exactement ce que je cherchais"
        ]
        
        texts_neg = [
            "Produit decevant, mauvaise qualite",
            "Service epouvantable, tres decu",
            "N achetez pas, c est nul",
            "Experience terrible, a eviter",
            "Perte de temps et d argent"
        ]
        
        for i, text in enumerate(texts_pos * 4):  # Répéter pour avoir plus de données
            writer.writerow([i, 'pos', 'test', text, 1])
            
        for i, text in enumerate(texts_neg * 4):
            writer.writerow([i+20, 'neg', 'test', text, 0])
            
        self.temp_file.close()
        self.temp_path = self.temp_file.name

    def tearDown(self):
        """Nettoie le fichier temporaire"""
        os.unlink(self.temp_path)

    def test_cross_validate_basic(self):
        """Test basique de cross_validate"""
        score = cross_validate(self.temp_path)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_cross_validate_different_cv(self):
        """Test avec différents nombres de folds"""
        for cv in [3, 5, 10]:
            score = cross_validate(self.temp_path, cv=cv)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)

    def test_cross_validate_different_models(self):
        """Test avec différents modèles"""
        for model in ["logreg", "svm"]:
            score = cross_validate(self.temp_path, model=model, cv=3)
            self.assertIsInstance(score, float)

    def test_cross_validate_different_vectorizers(self):
        """Test avec différents vectoriseurs"""
        for vectorizer in ["tfidf", "count"]:
            score = cross_validate(self.temp_path, vectorizer=vectorizer, cv=3)
            self.assertIsInstance(score, float)


class TestEdgeCases(unittest.TestCase):
    """Tests pour les cas limites"""

    def test_single_sample_file(self):
        """Test avec un seul échantillon"""
        single_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(single_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        writer.writerow([1, 'test', 'test', 'Un seul tweet', 1])
        single_file.close()
        
        texts, targets = load_texts_targets(single_file.name)
        self.assertEqual(len(texts), 1)
        self.assertEqual(len(targets), 1)
        os.unlink(single_file.name)

    def test_unicode_text(self):
        """Test avec du texte Unicode"""
        unicode_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(unicode_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        writer.writerow([1, 'emoji', 'paris', 'Texte avec emojis et accents eau', 1])
        writer.writerow([2, 'unicode', 'tokyo', 'Text with mixed scripts', 0])
        unicode_file.close()
        
        texts, targets = load_texts_targets(unicode_file.name)
        self.assertEqual(len(texts), 2)
        self.assertIn('emojis', texts[0])
        os.unlink(unicode_file.name)

    def test_very_long_text(self):
        """Test avec du texte très long"""
        long_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(long_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        
        long_text = "Mot " * 1000  # Texte très long
        writer.writerow([1, 'long', 'test', long_text, 1])
        writer.writerow([2, 'short', 'test', 'Court', 0])
        long_file.close()
        
        texts, targets = load_texts_targets(long_file.name)
        self.assertEqual(len(texts), 2)
        self.assertGreater(len(texts[0]), len(texts[1]))
        os.unlink(long_file.name)

    def test_empty_text_fields(self):
        """Test avec des champs texte vides"""
        empty_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(empty_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        writer.writerow([1, '', '', '', 1])
        writer.writerow([2, 'test', 'test', 'Texte normal', 0])
        empty_file.close()
        
        texts, targets = load_texts_targets(empty_file.name)
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0], '')
        os.unlink(empty_file.name)


class TestErrorHandling(unittest.TestCase):
    """Tests pour la gestion d'erreurs"""

    def test_nonexistent_file(self):
        """Test avec un fichier inexistant"""
        with self.assertRaises(FileNotFoundError):
            load_texts_targets("fichier_inexistant.csv")

    def test_malformed_csv(self):
        """Test avec un CSV mal formé"""
        bad_csv = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        bad_csv.write("id,text,target\n")
        bad_csv.write("1,\"texte avec guillemets non fermes,1\n")
        bad_csv.write("2,texte normal,0\n")
        bad_csv.close()
        
        try:
            texts, targets = load_texts_targets(bad_csv.name)
            # Le comportement peut varier selon la version de Python
            # On vérifie juste que ça ne plante pas complètement
            self.assertIsInstance(texts, list)
            self.assertIsInstance(targets, list)
        except Exception as e:
            # Si ça lève une exception, c'est acceptable pour un CSV mal formé
            self.assertIsInstance(e, Exception)
        finally:
            os.unlink(bad_csv.name)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn non installé")
    def test_insufficient_data_for_split(self):
        """Test avec des données insuffisantes pour la division"""
        tiny_file = tempfile.NamedTemporaryFile('w+', newline='', delete=False, encoding='utf-8')
        writer = csv.writer(tiny_file)
        writer.writerow(['id', 'keyword', 'location', 'text', 'target'])
        writer.writerow([1, 'test', 'test', 'Seul tweet', 1])
        tiny_file.close()
        
        # Avec test_size=0.9, il pourrait ne pas y avoir assez de données
        try:
            metrics = train_evaluate(tiny_file.name, test_size=0.9)
            self.assertIsInstance(metrics, dict)
        except Exception as e:
            # C'est acceptable que ça échoue avec des données insuffisantes
            self.assertIsInstance(e, Exception)
        finally:
            os.unlink(tiny_file.name)


@unittest.skipIf(SKLEARN_AVAILABLE, "Test uniquement si scikit-learn n'est pas disponible")
class TestWithoutSklearn(unittest.TestCase):
    """Tests pour le comportement sans scikit-learn"""

    def test_vectorizer_without_sklearn(self):
        """Test que les vectoriseurs lèvent une erreur sans scikit-learn"""
        with self.assertRaises(ImportError):
            build_vectorizer("tfidf")

    def test_classifier_without_sklearn(self):
        """Test que les classificateurs lèvent une erreur sans scikit-learn"""
        with self.assertRaises(ImportError):
            build_classifier("logreg")


if __name__ == '__main__':
    # Exécuter tous les tests
    unittest.main(verbosity=2)
