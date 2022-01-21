using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace AvgPerceptTest
{
    class Program
    {
        static void Main(string[] args)
        {
            bool oldway = false;
            bool complex = false;

            if (complex)
            {
                ComplexTest.RunComplexTest(oldway);
            }
            else
            {
                SimpleTest.RunSimpleTest(oldway);
            }
        }
    }
}
