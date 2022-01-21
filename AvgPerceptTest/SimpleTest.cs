using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AvgPerceptTest
{
    static class SimpleTest
    {
        public static void RunSimpleTest(bool oldway)
        {
            MLContext mlContext = new MLContext(seed: 0);

            List<TextLoader.Column> mlCols = new List<TextLoader.Column>();

            mlCols.Add(new TextLoader.Column("Stat1", DataKind.Single, 0));
            mlCols.Add(new TextLoader.Column("Result", DataKind.Boolean, 1));

            IDataView dataView = mlContext.Data.LoadFromTextFile("BC_AP CSV Data Simple.csv", mlCols.ToArray(), ',', true, true, true, false);

            var split = mlContext.Data.TrainTestSplit(dataView);

            IDataView trainingDataView = split.TrainSet;
            IDataView testingDataView = split.TestSet;

            IEstimator<ITransformer> pipeline = mlContext.Transforms.CopyColumns("Label", "Result");

            List<string> concatCols = new List<string>()
            {
                "Stat1"
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
                var predEngine = mlContext.Model.CreatePredictionEngine<SimpleDataObj, SimplePredDataObj>(trainedModel);

                SimpleDataObj dataObj = new SimpleDataObj(99);

                Console.WriteLine("Data for prediction - Expecting True");
                Console.WriteLine(dataObj.ToString());
                Console.WriteLine(string.Empty);

                var predData = predEngine.Predict(dataObj);

                Console.WriteLine("Result");
                Console.WriteLine(predData.Result);
            }
            else
            {
                List<SimpleDataObj> predData = new List<SimpleDataObj>()
                {
                    new SimpleDataObj(99)
                };

                Console.WriteLine("Data for prediction - Expecting True");
                Console.WriteLine(predData[0].ToString());
                Console.WriteLine(string.Empty);

                var mlPredData = mlContext.Data.LoadFromEnumerable(predData);
                var mlTransformedPredData = trainedModel.Transform(mlPredData);
                var apPredictions = mlContext.Data.CreateEnumerable<SimplePredDataObj>(mlTransformedPredData, false);

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

    public class SimpleDataObj
    {
        [LoadColumn(0)]
        public float Stat1 = 0;
        [LoadColumn(1)]
        public bool Result;

        public SimpleDataObj()
        {
        }

        public SimpleDataObj(float stat1)
        {
            Stat1 = stat1;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            sb.AppendLine("Stat1 = " + Stat1);

            return sb.ToString();
        }
    }

    public class SimplePredDataObj
    {
        [ColumnName("Result")]
        public bool Result { get; set; }
    }
}
