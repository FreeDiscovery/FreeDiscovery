
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Windows.Forms;

namespace FreeDiscoveryApp
{
    /// <summary>
    /// Form Main
    /// </summary>
    public partial class FormMain : Form
    {
        #region Constant

        private const string BASE_URL = "http://localhost:5001/";
        private readonly string[] ALGORITHM = { "LinearSVC", "LogisticRegression", "xgboost", "NearestCentroid", "NearestNeighbor" };

        #endregion

        #region Enum

        private enum ProcessType : int
        {
            Init = 0,
            Launch,
            RunAlgorithm,
            RunModel
        }

        #endregion

        #region Variables

        private ProcessType process;
        private FreeDiscovery.FreeDiscovery freeDiscovery;
        private string trainingFile;
        private string currentAlgorithm;

        #endregion

        #region Constructor

        /// <summary>
        /// Constructor
        /// </summary>
        public FormMain()
        {
            InitializeComponent();

            //Create FreeDiscovery Class
            freeDiscovery = new FreeDiscovery.FreeDiscovery(BASE_URL);
        }

        #endregion

        #region Event

        /// <summary>
        /// Load Event
        /// </summary>
        /// <param name="sender">Form</param>
        /// <param name="e">EventArgs</param>
        private void FormMain_Load(object sender, EventArgs e)
        {
            this.lblProcess.Text = string.Empty;
            this.grpAlgorithm.Enabled = false;

            //Set Init mode
            this.SetMode(ProcessType.Init);
        }

        /// <summary>
        /// Launch Event
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnLaunch_Click(object sender, EventArgs e)
        {
            if (bgWorker.IsBusy)
                return;

            //Set Launch mode
            this.SetMode(ProcessType.Launch);

            //Run
            bgWorker.RunWorkerAsync();
        }

        /// <summary>
        /// Browser Event
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnBrowser_Click(object sender, EventArgs e)
        {
            this.OpenFile();
            this.cmbAlgorithm.Enabled = true;
        }

        /// <summary>
        /// RunModel Event
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnRunModel_Click(object sender, EventArgs e)
        {
            if (bgWorker.IsBusy)
                return;

            //Set RunModel mode
            this.SetMode(ProcessType.RunModel);

            //Run
            bgWorker.RunWorkerAsync();
        }

        /// <summary>
        /// SelectedIndexChanged Event
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cmbAlgorithm_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (bgWorker.IsBusy)
                return;

            //Get Algorithm
            Algorithm item = (Algorithm)this.cmbAlgorithm.SelectedItem;

            //Check Algorithm
            if (item.Code != 0)
            {
                currentAlgorithm = ALGORITHM[this.cmbAlgorithm.SelectedIndex - 1];

                //Set RunAlgorithm mode
                this.SetMode(ProcessType.RunAlgorithm);

                //Run
                bgWorker.RunWorkerAsync();
            }
        }

        /// <summary>
        /// DoWork Event
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void bgWorker_DoWork(object sender, DoWorkEventArgs e)
        {            
            //Run
            this.DoWork(this.process, e);
        }

        /// <summary>
        /// RunWorkerCompleted Event
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void bgWorker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            //Launch mode case
            switch (this.process)
            {
                case ProcessType.Launch:

                    this.SetMode(ProcessType.Init);

                    if ((bool)e.Result != true)
                    {
                        MessageBox.Show("There was an error connecting to " + BASE_URL, this.Text);
                    }
                    else
                    {
                        this.grpAlgorithm.Enabled = true;
                        this.btnRunModel.Enabled = false;
                        this.cmbAlgorithm.Enabled = false;
                        this.btnLaunch.Enabled = false;

                        this.InitAlgorithm();
                    }

                    break;

                case ProcessType.RunAlgorithm:

                    this.SetMode(ProcessType.Init);
                    FreeDiscovery.ResultInfo resultInfo = (FreeDiscovery.ResultInfo)e.Result;

                    if (resultInfo.Result != true)
                    {
                        MessageBox.Show(resultInfo.Error, this.Text);
                    }
                    else
                    {
                        this.btnRunModel.Enabled = true;
                    }

                    break;

                case ProcessType.RunModel:

                    this.SetMode(ProcessType.Init);

                    this.lblProcess.Text = string.Format("Your model is {0}% accurate", e.Result);

                    break;
            }            
        }

        #endregion        

        #region Method

        /// <summary>
        /// Set Mode
        /// </summary>
        /// <param name="mode">Mode</param>
        private void SetMode(ProcessType mode)
        {
            this.process = mode;

            switch (this.process)
            {
                case ProcessType.Init:
                    this.lblProcess.Text = string.Empty;
                    this.progressBar1.Style = ProgressBarStyle.Blocks;
                    break;
                case ProcessType.Launch:
                    this.lblProcess.Text = "SERVICE UP AND RUNNING";
                    this.progressBar1.Style = ProgressBarStyle.Marquee;
                    break;
                case ProcessType.RunAlgorithm:
                    this.lblProcess.Text = "PLEASE WAIT WHILE FREE DISCOVERY LEARNS A MODEL";
                    this.progressBar1.Style = ProgressBarStyle.Marquee;
                    break;
                case ProcessType.RunModel:
                    this.lblProcess.Text = "RUN MODEL";
                    this.progressBar1.Style = ProgressBarStyle.Marquee;
                    break;
            }
        }

        /// <summary>
        /// Run
        /// </summary>
        /// <param name="mode">Mode</param>
        private void DoWork(ProcessType mode, DoWorkEventArgs e)
        {
            switch (mode)
            {
                case ProcessType.Launch:

                    e.Result = freeDiscovery.CheckServerConnection();

                    break;

                case ProcessType.RunAlgorithm:

                    e.Result = freeDiscovery.RunAlgorithm(this.trainingFile, this.currentAlgorithm);

                    break;

                case ProcessType.RunModel:

                    e.Result = freeDiscovery.GetAccurate();

                    break;
            }
            
        }

        /// <summary>
        /// Init Algorithm
        /// </summary>
        private void InitAlgorithm()
        {
            List<Algorithm> list = new List<Algorithm>();

            Algorithm a0 = new Algorithm();
            a0.Code = 0;
            a0.Name = "---";

            Algorithm a1 = new Algorithm();
            a1.Code = 1;
            a1.Name = "Linear SVC";

            Algorithm a2 = new Algorithm();
            a2.Code = 2;
            a2.Name = "Logistic Regression";

            Algorithm a3 = new Algorithm();
            a3.Code = 3;
            a3.Name = "xgboost";

            Algorithm a4 = new Algorithm();
            a4.Code = 4;
            a4.Name = "Nearest Centroid";

            Algorithm a5 = new Algorithm();
            a5.Code = 5;
            a5.Name = "Nearest Neighbor";

            list.Add(a0);
            list.Add(a1);
            list.Add(a2);
            list.Add(a3);
            list.Add(a4);
            list.Add(a5);

            this.cmbAlgorithm.DisplayMember = "Name";
            this.cmbAlgorithm.ValueMember = "Code";
            this.cmbAlgorithm.DataSource = list;
        }

        private void OpenFile()
        {
            OpenFileDialog theDialog = new OpenFileDialog();
            theDialog.InitialDirectory = @"C:\";
            theDialog.Filter = "CSV file (*.csv)|*.csv";

            if (theDialog.ShowDialog() == DialogResult.OK)
            {
                this.txtCSVFile.Text = theDialog.FileName;
                this.trainingFile = this.txtCSVFile.Text;
            }
        }

        #endregion
    }
}
