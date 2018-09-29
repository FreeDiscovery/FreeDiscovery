using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Windows.Forms;

namespace FreeDiscovery
{
    /// <summary>
    /// FreeDiscovery Class
    /// </summary>
    public class FreeDiscovery
    {
        #region Constant

        private const string BASE_URL = "http://localhost:5001/";
        private const string FOLDER_PROCESSING = "dataset";
        private const int FOLD = 10;

        #endregion

        #region Variables

        private string algorithm;
        private string trainingFile;
        private string baseUrl;
        private HttpClient client;

        private Dictionary<int, DataInfo> dataTrain;
        private int dataTrainCount;
        
        private string FD_DATASET_ID;
        private string FD_LSI_ID;

        #endregion

        #region Constructor

        public FreeDiscovery()
        {
            baseUrl = BASE_URL;

            client = new HttpClient();
            client.BaseAddress = new Uri(BASE_URL);
            client.DefaultRequestHeaders.Accept.Clear();
            client.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json"));
        }

        public FreeDiscovery(string url)
        {
            baseUrl = url;

            client = new HttpClient();
            client.BaseAddress = new Uri(baseUrl);
            client.DefaultRequestHeaders.Accept.Clear();
            client.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json"));
        }

        #endregion

        #region Method

        /// <summary>
        /// Checking server connection
        /// </summary>
        /// <returns>Success:true,orlese:false</returns>
        public bool CheckServerConnection()
        {
            bool ret = true;
            try
            {
                var responce = client.GetAsync("").Result;
            }
            catch (Exception e)
            {
                ret = false;
            }
            return ret;
        }

        /// <summary>
        /// Run Algorithm
        /// </summary>
        /// <param name="file">Training CSV File</param>
        /// <param name="algorithm">Algorithm</param>
        /// <returns></returns>
        public ResultInfo RunAlgorithm(string file, string algorithm)
        {
            ResultInfo ret = new ResultInfo();
            ret.Result = true;

            try
            {
                this.trainingFile = file;
                this.algorithm = algorithm;

                //Init
                this.Init();

                //Read training data
                this.ReadTrainingData();

                //divide the data into 10 pieces
                this.DivideTrainingData();

                //Create dataset for training
                this.CreateDataSet();

                //1. Feature extraction

                //1.a Load dataset and initalize feature extraction
                JObject jo = PostAsync("/api/v0/feature-extraction", new { });
                this.FD_DATASET_ID = jo["id"].ToString();

                //1b. Start feature extraction
                jo = PostAsync(string.Format("/api/v0/feature-extraction/{0}", FD_DATASET_ID), new { data_dir = Path.Combine(Application.StartupPath, FOLDER_PROCESSING) });
                if (jo["messages"] != null)
                {
                    ret.Error = jo["messages"].ToString();
                    ret.Result = false;
                    return ret;
                }

                //2. Calculate Latent Semantic Indexing
                jo = PostAsync("/api/v0/lsi/", new { parent_id = FD_DATASET_ID });
                if (jo["messages"] != null)
                {
                    ret.Error = jo["messages"].ToString();
                    ret.Result = false;
                    return ret;
                }
                FD_LSI_ID = jo["id"].ToString();

                //10fold cross validation
                for (int i = 0; i < FOLD; i++)
                {
                    //Get Train Data
                    dynamic dataCategory = this.GetTrainData(i);

                    //3.a Create a categorization model
                    jo = PostAsync("/api/v0/categorization/", new
                    {
                        parent_id = FD_DATASET_ID,
                        method = this.algorithm,
                        data = dataCategory,
                        training_scores = true
                    });

                    if (jo["messages"] != null)
                    {
                        continue;
                    }

                    //3.b Predictions for the other documents in the dataset
                    jo = GetAsync(string.Format("/api/v0/categorization/{0}/predict", jo["id"].ToString()));
                    if (jo["messages"] != null)
                    {
                        continue;
                    }

                    //Predictions
                    var predictionsCsv =
                    from p in jo["data"]
                    select new { document_id = (string)p["document_id"], category = (string)p["scores"][0]["category"] };
                    foreach (var item in predictionsCsv)
                    {
                        DataInfo info = this.dataTrain[Convert.ToInt32(item.document_id)];
                        info.Prediction = item.category;
                    }
                }

                //5. Delete the extracted features (and LSI decomposition)
                var responce = client.DeleteAsync(string.Format("/api/v0/feature-extraction/{0}", FD_DATASET_ID)).Result;

                //6. Create Model.csv
                this.CreateModelCSV();
            }
            catch(Exception e)
            {
                ret.Error = e.Message;
                ret.Result = false;
                return ret;
            }

            return ret;
        }

        /// <summary>
        /// Get accurate
        /// </summary>
        /// <returns>Accurate</returns>
        public string GetAccurate()
        {
            int countRight = 0;
            foreach (int key in this.dataTrain.Keys)
            {
                DataInfo info = this.dataTrain[key];
                if(info.Category == info.Prediction)
                {
                    countRight++;
                }
            }

            return (countRight * 100 / this.dataTrainCount).ToString();
        }

        #endregion

        #region Private Method

        /// <summary>
        /// Sends a POST request to the specified Uri as an asynchronous operation.
        /// </summary>
        /// <param name="url">The Uri the request is sent to.</param>
        /// <param name="data">The HTTP request content sent to the server.</param>
        /// <returns>The object representing the asynchronous operation.</returns>
        private JObject PostAsync(string url, object data)
        {
            HttpResponseMessage responce = client.PostAsync(url, CreateJsonData(data)).Result;
            return JObject.Parse(responce.Content.ReadAsStringAsync().Result);
        }

        /// <summary>
        /// Sends a GET request to the specified Uri as an asynchronous operation.
        /// </summary>
        /// <param name="url">The Uri the request is sent to.</param>
        /// <param name="data">The HTTP request content sent to the server.</param>
        /// <returns>The object representing the asynchronous operation.</returns>
        private JObject GetAsync(string url)
        {
            HttpResponseMessage responce = client.GetAsync(url).Result;
            return JObject.Parse(responce.Content.ReadAsStringAsync().Result);
        }

        /// <summary>
        /// Create Json Data
        /// </summary>
        /// <param name="data">data</param>
        /// <returns>StringContent</returns>
        private StringContent CreateJsonData(object data)
        {
            string json = JsonConvert.SerializeObject(data);
            return new StringContent(json, UnicodeEncoding.UTF8, "application/json");
        }

        private void Init()
        {
            this.dataTrain = new Dictionary<int, DataInfo>();
            this.dataTrainCount = 0;

            //Create folder for processing logic
            string path = Path.Combine(Application.StartupPath, FOLDER_PROCESSING);
            Directory.CreateDirectory(path);

            //Delete all files
            DirectoryInfo di = new DirectoryInfo(path);
            foreach (FileInfo file in di.GetFiles())
            {
                file.Delete();
            }
        }

        /// <summary>
        /// Read Train Data
        /// </summary>
        private void ReadTrainingData()
        {
            int i = this.dataTrain.Count;
            using (var rd = new StreamReader(this.trainingFile))
            {
                while (!rd.EndOfStream)
                {
                    string contain = rd.ReadLine();
                    string[] val = contain.Split(',');

                    if (val.Length > 0)
                    {
                        DataInfo info = new DataInfo();
                        info.DocumentId = i;
                        info.FilePath = i.ToString().PadLeft(10, '0');
                        info.Contain = contain;
                        info.GroupData = -1;

                        if (val[val.Length - 1].Replace("\"", "") == "Y")
                        {
                            info.Category = "Y";
                        }
                        else
                        {
                            info.Category = "N";
                        }

                        this.dataTrain.Add(i, info);
                        this.dataTrainCount++;
                        i++;
                    }
                }
            }
        }

        /// <summary>
        /// Divide the data into FOLDS pieces
        /// </summary>
        private void DivideTrainingData()
        {
            //Create fold
            int groupY = 0;
            int groupN = 0;
            Dictionary<int, int> listFold = new Dictionary<int, int>();
            for (int l = 0; l < FOLD; l++)
            {
                listFold.Add(l, 0);
            }

            foreach (int key in this.dataTrain.Keys)
            {
                if (groupY == FOLD && groupN == FOLD)
                {
                    break;
                }

                DataInfo info = this.dataTrain[key];
                if (info.GroupData == -1)
                {
                    if (info.Category == "Y" && groupY < FOLD)
                    {
                        info.GroupData = groupY;
                        listFold[groupY] += 1;
                        groupY++;
                    }

                    if (info.Category == "N" && groupN < FOLD)
                    {
                        info.GroupData = groupN;
                        listFold[groupN] += 1;
                        groupN++;
                    }
                }
            }

            int fold = this.dataTrainCount / FOLD;
            int j = 0;
            foreach (int key in this.dataTrain.Keys)
            {
                DataInfo info = this.dataTrain[key];
                if (info.GroupData == -1)
                {
                    if (listFold[j] >= fold && j < FOLD - 1)
                    {
                        j++;
                    }
                    this.dataTrain[key].GroupData = j;
                    listFold[j] += 1;
                }
            }
        }

        /// <summary>
        /// Create dataset
        /// </summary>
        private void CreateDataSet()
        {
            string path = Path.Combine(Application.StartupPath, FOLDER_PROCESSING);
            foreach (int key in this.dataTrain.Keys)
            {
                DataInfo info = this.dataTrain[key];
                if (info.GroupData != -1)
                {
                    StringBuilder csv = new StringBuilder();
                    string[] data = dataTrain[key].Contain.Split(',');
                    data = data.Where(w => w != data[data.Length -1]).ToArray();                    
                    csv.Append(string.Join(",", data));
                    string fileName = Path.Combine(path, info.FilePath);
                    File.WriteAllText(fileName, csv.ToString());
                }
            }
        }

        /// <summary>
        /// Create Model.csv
        /// </summary>
        private void CreateModelCSV()
        {
            StringBuilder csv = new StringBuilder();
            foreach (int key in this.dataTrain.Keys)
            {
                DataInfo info = this.dataTrain[key];
                csv.AppendLine(dataTrain[key].Contain + ",\"" + dataTrain[key].Prediction + "\"");
            }

            string fileName = Path.Combine(Application.StartupPath, "Model.csv");
            File.WriteAllText(fileName, csv.ToString());
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="groupData"></param>
        /// <returns></returns>
        private List<dynamic> GetTrainData(int groupData)
        {
            dynamic ret = new List<dynamic>();

            foreach (int key in this.dataTrain.Keys)
            {
                DataInfo info = this.dataTrain[key];
                if (info.GroupData != groupData)
                {
                    ret.Add(new { document_id = key, category = info.Category });
                }
            }

            return ret;
        }

        #endregion
    }
}
