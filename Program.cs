using System;
using System.Drawing;

namespace Réseau_de_neurones
{
    class Program
    {
        static void Main(string[] args)
        {
            Réseau r = new Réseau();
            int[] inputData;
            for (int i = 0; i <= 13; i++)
            {
                inputData = GetDataFromImages(i);
                r.Propager(inputData);
                r.Apprendre();
                Console.WriteLine(r);
            }
            inputData = GetDataFromImages(14);
            r.Propager(inputData);
            Console.WriteLine(">>> PHASE DE TEST <<<");
            Console.WriteLine(r);
            // Comptage du nombre de pixels noirs pour afficher les données souhaitées sur l'image de test
            byte nbPixelsNoirs = 0;
            foreach (int pixel in inputData)
                nbPixelsNoirs += Convert.ToByte(pixel);
            Console.WriteLine($"Sortie souhaitée : {Convert.ToString(nbPixelsNoirs, 2)}"); //Valeurs possibles : 00 01 10 11 (conversion en binaire)
        }

        /// <summary>
        /// Lecture de l'image
        /// Extraction des pixels
        /// Conversion en valeur binaire (1 pour noir et 0 pour blanc)
        /// </summary>
        /// <param name="imageNum">Le numéro de l'image à lire</param>
        /// <returns>Tableau contenant les données des pixels</returns>
        private static int[] GetDataFromImages(int imageNum)
        {
            Bitmap image = new Bitmap($"..\\..\\..\\images\\frame-{ imageNum }.png");
            int[] data = new int[] { 0, 0, 0, 0 };
            int k = 0;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                {
                    Color pixel = image.GetPixel(j, i);
                    //Console.WriteLine(pixel);
                    // 1 si le pixel est noir, 0 sinon
                    data[k++] = pixel == Color.FromArgb(255, 0, 0, 0) ? 1 : 0;
                }
            return data;
        }
    }
}
