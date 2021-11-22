using System;

namespace Réseau_de_neurones
{
    class Réseau
    {
        /// <summary>
        /// Données d'entrée
        /// </summary>
        private int[] Input { get; set; }
        /// <summary>
        /// Données dans la couche cachée
        /// </summary>
        private double[] Hidden { get; set; }
        /// <summary>
        /// Données de sortie
        /// </summary>
        private double[] Output { get; set; }
        /// <summary>
        /// Poids des connexions reliant la couche d'entrée et la cachée
        /// </summary>
        private double[,] wHidden { get; set; }
        /// <summary>
        /// Poids des connexions reliant la couche cachée et celle de sortie
        /// </summary>
        private double[,] wOutput { get; set; }
        /// <summary>
        /// Erreurs dans le réseau en sortie
        /// </summary>
        private double[] Error { get; set; }
        /// <summary>
        /// Taux d'apprentissage
        /// </summary>
        private double Alpha { get; set; }
        /// <summary>
        /// Gradients d'erreurs de la couche de sortie
        /// </summary>
        private double[,] wOutputGradient { get; set; }
        /// <summary>
        /// Gradients d'erreur de la couche cachée
        /// </summary>
        private double[,] wHidddenGradient { get; set; }

        /// <summary>
        /// Constructeur initialisant le réseau avec des valeurs arbitraires
        /// </summary>
        public Réseau()
        {
            Input = new int[] { 0, 0, 0, 0 };
            Hidden = new double[] { 0, 0, 0, 0 };
            Output = new double[] { 0, 0 };
            Error = new double[2];
            wHidden = new double[4, 4];
            wOutput = new double[2, 4];
            wOutputGradient = new double[2, 4];
            wHidddenGradient = new double[4, 4];
            InitPoids(wHidden, 0.5);
            InitPoids(wOutput, 0.5);
        }

        /// <summary>
        /// Initialise les poids arbitrairement à 0.5
        /// </summary>
        /// <param name="tab">Le tableau contenant les poids entre 2 couches</param>
        private void InitPoids(double[,] tab, double value)
        {
            for (int n = 0; n < tab.GetLength(0); n++)
                for (int m = 0; m < tab.GetLength(1); m++)
                    tab[n, m] = value;
        }

        /// <summary>
        /// Calcule la valeur de sortie d'un neurone
        /// </summary>
        /// <param name="x">poids du neurone</param>
        /// <returns>La valeur de sortie d'un neurone</returns>
        private double Sigmoide(double x)
            => 1 / (1 + Math.Exp(-x));

        public void Apprendre()
        {
            Alpha = 0.5; // Taux d'apprentissage
            int[] Target = new int[] { 0, 0 }; // Données voulues à l'issu de l'apprentissage

            // Calcul de l'erreur en sortie, après avoir propagé
            for (int k = 0; k < Output.Length; k++)
                Error[k] = Target[k] - Output[k];

            // Calcul des gradients d'erreur de la couche de sortie Output
            InitPoids(wOutputGradient, 0);
            for (var k = 0; k < Output.Length; k++)
                for (var j = 0; j < Hidden.Length; j++)
                    wOutputGradient[k, j] = -Error[k] * Output[k] * (1 - Output[k]) * Hidden[j];

            // Calcul des gradients d'erreur de la couche cachée Hidden
            // Avec rétropropagation de l'erreur de Output vers Hidden pour l'affecter aux neurones
            InitPoids(wHidddenGradient, 0);
            for (var j = 0; j < Hidden.Length; j++)
                for (var i = 0; i < Input.Length; i++)
                {
                    double e = 0;
                    for (var k = 0; k < Output.Length; k++)
                        e += wOutput[k, j] * Error[k];
                    wHidddenGradient[j, i] = -e * Hidden[j] * (1 - Hidden[j]) * Input[i];
                }

            // Mise à jour de tous les poids du réseau
            Update();
        }

        /// <summary>
        /// Propage les données de la couche d'entrée vers la couche de sortie
        /// </summary>
        public void Propager(int[] inputData)
        {
            // Copie des données d'entrée dans la couche d'entrée
            Array.Copy(inputData, Input, Input.Length);

            // Tableau intermédiaire servant pour appliquer la fonction Sigmoide sur celui-ci
            // Et ensuite l'affecter au tableau Hidden
            // On a donc des valeurs actualisées
            double[] tempHidden = new double[] { 0, 0, 0, 0 };
            for (int j = 0; j < Hidden.Length; j++)
            {
                for (int i = 0; i < Input.Length; i++)
                    tempHidden[j] += wHidden[j, i] * Input[i];
                Hidden[j] = Sigmoide(tempHidden[j]);
            }

            // Même chose que précédemment, mais ce tableau servira pour les neurones de sortie
            double[] tempOutput = new double[] { 0, 0 };
            for (int k = 0; k < Output.Length; k++)
            {
                for (int j = 0; j < Hidden.Length; j++)
                    tempOutput[k] += wOutput[k, j] * Hidden[j];
                Output[k] = Sigmoide(tempOutput[k]);
            }
        }

        public override string ToString()
        {
            string str = $"Entrées : {Input[0]} {Input[1]} {Input[2]} {Input[3]}\n";
            str += $"Sortie 1 : {Output[0]}\nSortie 2 : {Output[1]}\n\n";
            str += $"Erreur 1 : {Error[0]}\nErreur 2 : {Error[1]}\n\n";
            return str;
        }

        /// <summary>
        /// Mise à jour de tous les poids du réseau
        /// </summary>
        private void Update()
        {
            // Couche de sortie
            for (var k = 0; k < Output.Length; k++)
                for (var j = 0; j < Hidden.Length; j++)
                    wOutput[k, j] -= Alpha * wOutputGradient[k, j];

            // Couche cachée
            for (var j = 0; j < Hidden.Length; j++)
                for (var i = 0; i < Input.Length; i++)
                    wHidden[j, i] -= Alpha * wHidddenGradient[j, i];
        }
    }
}
