using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Regression_TaxiFarePrediction.DataStructures;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Windows.AI.MachineLearning;
using Windows.ApplicationModel.Core;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.UI;
using Windows.UI.ViewManagement;
using Windows.UI.Xaml.Controls;

namespace XamlBrewer.UWP.WinML.Sample
{
    public sealed partial class MainPage : Page
    {
        public MainPage()
        {
            var coreTitleBar = CoreApplication.GetCurrentView().TitleBar;
            coreTitleBar.ExtendViewIntoTitleBar = true;

            var titleBar = ApplicationView.GetForCurrentView().TitleBar;
            if (titleBar != null)
            {
                titleBar.BackgroundColor = Colors.Transparent;
                titleBar.ButtonBackgroundColor = Colors.Transparent;
                titleBar.ButtonInactiveBackgroundColor = Colors.SlateGray;
                titleBar.ButtonForegroundColor = (Color)Resources["SystemAccentColor"];
            }

            this.InitializeComponent();
        }

        private async void MLNet_Button_Click(object sender, Windows.UI.Xaml.RoutedEventArgs e)
        {
            ResultTextBlock.Text = string.Empty;

            var trip = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripDistance = 10.33f,
                PaymentType = "CSH",
                FareAmount = 0 // predict it. actual = 29.5
            };

            MLContext mlContext = new MLContext(seed: 0);

            var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/TaxiFareModel.zip"));
            ITransformer trainedModel = mlContext.Model.Load(await modelFile.OpenStreamForReadAsync(), out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(trainedModel);
            var resultprediction = predEngine.Predict(trip);

            ResultTextBlock.Text = resultprediction.FareAmount.ToString("C");
        }

        private void Onnx_Button_Click(object sender, Windows.UI.Xaml.RoutedEventArgs e)
        {
            // var session = new InferenceSession("Assets/mnist.onnx");
            var session = new InferenceSession("Assets/TaxiFareModel.onnx");
            var inputMeta = session.InputMetadata;
            var container = new List<NamedOnnxValue>
            {
                GetNamedOnnxValue<string>(inputMeta, "VendorId", "VTS"),
                GetNamedOnnxValue<string>(inputMeta, "RateCode", "1"),
                GetNamedOnnxValue<float>(inputMeta, "PassengerCount", 1f),
                GetNamedOnnxValue<float>(inputMeta, "TripDistance", 10.33f),
                GetNamedOnnxValue<string>(inputMeta, "PaymentType", "CSH")
            };

            var result = session.Run(container);
            var output = result.First(x => x.Name == "Score0").AsTensor<float>().Max();
        }

        private static NamedOnnxValue GetNamedOnnxValue<T>(IReadOnlyDictionary<string, NodeMetadata> inputMeta, string column, T value)
        {
            T[] inputDataInt = new T[] { value };
            var tensor = new DenseTensor<T>(inputDataInt, inputMeta[column].Dimensions);
            var namedOnnxValue = NamedOnnxValue.CreateFromTensor<T>(column, tensor);
            namedOnnxValue.Value = value;
            return namedOnnxValue;
        }

        private async void WinML_Button_Click(object sender, Windows.UI.Xaml.RoutedEventArgs e)
        {
            //var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/mnist.onnx"));
            var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/TaxiFareModel.onnx"));
            var learningModel = await LearningModel.LoadFromStreamAsync(modelFile);
        }
    }
}
