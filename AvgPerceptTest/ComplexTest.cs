using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AvgPerceptTest
{
    static class ComplexTest
    {
        public static void RunComplexTest(bool oldway)
        {
            MLContext mlContext = new MLContext(seed: 0);

            List<TextLoader.Column> mlCols = new List<TextLoader.Column>();

            mlCols.Add(new TextLoader.Column("Stat1", DataKind.Single, 0));
            mlCols.Add(new TextLoader.Column("Stat2", DataKind.Single, 1));
            mlCols.Add(new TextLoader.Column("Stat3", DataKind.Single, 2));
            mlCols.Add(new TextLoader.Column("Stat4", DataKind.Single, 3));
            mlCols.Add(new TextLoader.Column("Stat5", DataKind.Single, 4));
            mlCols.Add(new TextLoader.Column("Stat6", DataKind.Single, 5));
            mlCols.Add(new TextLoader.Column("Stat7", DataKind.Single, 6));
            mlCols.Add(new TextLoader.Column("Stat8", DataKind.Single, 7));
            mlCols.Add(new TextLoader.Column("Stat9", DataKind.Single, 8));
            mlCols.Add(new TextLoader.Column("Stat10", DataKind.Single, 9));
            mlCols.Add(new TextLoader.Column("Result", DataKind.Boolean, 1));

            IDataView dataView = mlContext.Data.LoadFromTextFile("BC_AP CSV Data Simple.csv", mlCols.ToArray(), ',', true, true, true, false);

            var split = mlContext.Data.TrainTestSplit(dataView);

            IDataView trainingDataView = split.TrainSet;
            IDataView testingDataView = split.TestSet;

            IEstimator<ITransformer> pipeline = mlContext.Transforms.CopyColumns("Label", "Result");

            List<string> concatCols = new List<string>()
            {
                 "Stat1"
                ,"Stat2"
                ,"Stat3"
                ,"Stat4"
                ,"Stat5"
                ,"Stat6"
                ,"Stat7"
                ,"Stat8"
                ,"Stat9"
                ,"Stat10"
            };

            pipeline = pipeline.Append(mlContext.Transforms.Concatenate("Features", concatCols.ToArray()));
            pipeline = pipeline.Append(mlContext.Transforms.NormalizeMeanVariance("Features", "Features"));

            var trainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron();

            var trainingPipeline = pipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            IDataView predictions = trainedModel.Transform(testingDataView);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(predictions);

            Console.WriteLine($"*Metrics for {trainer.ToString()} classifier model");
            Console.WriteLine(string.Empty);
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " + $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " + $"{metrics.PositivePrecision:F2}");

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine(string.Empty);

            if (oldway)
            {
                var predEngine = mlContext.Model.CreatePredictionEngine<DataObj, PredDataObj>(trainedModel);

                DataObj dataObj = new DataObj(99, 7, 14, 8, 38, 50, 23, 38, 28, 65);

                Console.WriteLine("Data for prediction - Expecting True");
                Console.WriteLine(dataObj.ToString());
                Console.WriteLine(string.Empty);

                var predData = predEngine.Predict(dataObj);

                Console.WriteLine("Result");
                Console.WriteLine(predData.Result);
            }
            else
            {
                List<DataObj> predData = new List<DataObj>()
                {
                    new DataObj(99, 7, 14, 8, 38, 50, 23, 38, 28, 65)
                };

                Console.WriteLine("Data for prediction - Expecting True");
                Console.WriteLine(predData[0].ToString());
                Console.WriteLine(string.Empty);

                var mlPredData = mlContext.Data.LoadFromEnumerable(predData);
                var mlTransformedPredData = trainedModel.Transform(mlPredData);
                var apPredictions = mlContext.Data.CreateEnumerable<PredDataObj>(mlTransformedPredData, false);

                Console.WriteLine("Result");

                using (var sequenceEnum = apPredictions.GetEnumerator())
                {
                    while (sequenceEnum.MoveNext())
                    {
                        Console.WriteLine(sequenceEnum.Current.Result);
                    }
                }
            }
        }
    }

    public class DataObj
    {
        [LoadColumn(0)]
        public float Stat1 = 0;
        [LoadColumn(1)]
        public float Stat2 = 0;
        [LoadColumn(2)]
        public float Stat3 = 0;
        [LoadColumn(3)]
        public float Stat4 = 0;
        [LoadColumn(4)]
        public float Stat5 = 0;
        [LoadColumn(5)]
        public float Stat6 = 0;
        [LoadColumn(6)]
        public float Stat7 = 0;
        [LoadColumn(7)]
        public float Stat8 = 0;
        [LoadColumn(8)]
        public float Stat9 = 0;
        [LoadColumn(9)]
        public float Stat10 = 0;
        [LoadColumn(1)]
        public bool Result;

        public DataObj()
        {
        }

        public DataObj(float stat1, float stat2, float stat3, float stat4, float stat5, float stat6, float stat7, float stat8, float stat9, float stat10)
        {
            Stat1 = stat1;
            Stat2 = stat2;
            Stat3 = stat3;
            Stat4 = stat4;
            Stat5 = stat5;
            Stat6 = stat6;
            Stat7 = stat7;
            Stat8 = stat8;
            Stat9 = stat9;
            Stat10 = stat10;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.AppendLine("Stat1 = " + Stat1);
            sb.AppendLine("Stat2 = " + Stat2);
            sb.AppendLine("Stat3 = " + Stat3);
            sb.AppendLine("Stat4 = " + Stat4);
            sb.AppendLine("Stat5 = " + Stat5);
            sb.AppendLine("Stat6 = " + Stat6);
            sb.AppendLine("Stat7 = " + Stat7);
            sb.AppendLine("Stat8 = " + Stat8);
            sb.AppendLine("Stat9 = " + Stat9);
            sb.AppendLine("Stat10 = " + Stat10);

            return sb.ToString();
        }
    }

    public class PredDataObj
    {
        [ColumnName("Result")]
        public bool Result { get; set; }
    }
}
