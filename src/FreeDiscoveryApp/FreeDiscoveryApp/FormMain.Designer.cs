namespace FreeDiscoveryApp
{
    partial class FormMain
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.btnLaunch = new System.Windows.Forms.Button();
            this.cmbAlgorithm = new System.Windows.Forms.ComboBox();
            this.btnBrowser = new System.Windows.Forms.Button();
            this.txtCSVFile = new System.Windows.Forms.TextBox();
            this.btnRunModel = new System.Windows.Forms.Button();
            this.progressBar1 = new System.Windows.Forms.ProgressBar();
            this.grpProcessing = new System.Windows.Forms.GroupBox();
            this.lblProcess = new System.Windows.Forms.Label();
            this.bgWorker = new System.ComponentModel.BackgroundWorker();
            this.grpAlgorithm = new System.Windows.Forms.GroupBox();
            this.lblAlgorithm = new System.Windows.Forms.Label();
            this.lblCSVFile = new System.Windows.Forms.Label();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.grpProcessing.SuspendLayout();
            this.grpAlgorithm.SuspendLayout();
            this.SuspendLayout();
            // 
            // btnLaunch
            // 
            this.btnLaunch.Location = new System.Drawing.Point(12, 149);
            this.btnLaunch.Name = "btnLaunch";
            this.btnLaunch.Size = new System.Drawing.Size(198, 235);
            this.btnLaunch.TabIndex = 0;
            this.btnLaunch.Text = "Launch";
            this.btnLaunch.UseVisualStyleBackColor = true;
            this.btnLaunch.Click += new System.EventHandler(this.btnLaunch_Click);
            // 
            // cmbAlgorithm
            // 
            this.cmbAlgorithm.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbAlgorithm.FormattingEnabled = true;
            this.cmbAlgorithm.Location = new System.Drawing.Point(88, 102);
            this.cmbAlgorithm.Name = "cmbAlgorithm";
            this.cmbAlgorithm.Size = new System.Drawing.Size(198, 21);
            this.cmbAlgorithm.TabIndex = 1;
            this.cmbAlgorithm.SelectedIndexChanged += new System.EventHandler(this.cmbAlgorithm_SelectedIndexChanged);
            // 
            // btnBrowser
            // 
            this.btnBrowser.Location = new System.Drawing.Point(485, 71);
            this.btnBrowser.Name = "btnBrowser";
            this.btnBrowser.Size = new System.Drawing.Size(75, 22);
            this.btnBrowser.TabIndex = 2;
            this.btnBrowser.Text = "Browser";
            this.btnBrowser.UseVisualStyleBackColor = true;
            this.btnBrowser.Click += new System.EventHandler(this.btnBrowser_Click);
            // 
            // txtCSVFile
            // 
            this.txtCSVFile.Location = new System.Drawing.Point(88, 72);
            this.txtCSVFile.Name = "txtCSVFile";
            this.txtCSVFile.ReadOnly = true;
            this.txtCSVFile.Size = new System.Drawing.Size(391, 20);
            this.txtCSVFile.TabIndex = 3;
            // 
            // btnRunModel
            // 
            this.btnRunModel.Location = new System.Drawing.Point(87, 134);
            this.btnRunModel.Name = "btnRunModel";
            this.btnRunModel.Size = new System.Drawing.Size(75, 23);
            this.btnRunModel.TabIndex = 4;
            this.btnRunModel.Text = "Run Model";
            this.btnRunModel.UseVisualStyleBackColor = true;
            this.btnRunModel.Click += new System.EventHandler(this.btnRunModel_Click);
            // 
            // progressBar1
            // 
            this.progressBar1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.progressBar1.Location = new System.Drawing.Point(6, 72);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(839, 23);
            this.progressBar1.TabIndex = 5;
            // 
            // grpProcessing
            // 
            this.grpProcessing.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.grpProcessing.Controls.Add(this.lblProcess);
            this.grpProcessing.Controls.Add(this.progressBar1);
            this.grpProcessing.Location = new System.Drawing.Point(12, 12);
            this.grpProcessing.Name = "grpProcessing";
            this.grpProcessing.Size = new System.Drawing.Size(851, 119);
            this.grpProcessing.TabIndex = 6;
            this.grpProcessing.TabStop = false;
            this.grpProcessing.Text = "Processing";
            // 
            // lblProcess
            // 
            this.lblProcess.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lblProcess.Location = new System.Drawing.Point(6, 40);
            this.lblProcess.Name = "lblProcess";
            this.lblProcess.Size = new System.Drawing.Size(839, 29);
            this.lblProcess.TabIndex = 6;
            this.lblProcess.Text = "label1";
            this.lblProcess.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // bgWorker
            // 
            this.bgWorker.DoWork += new System.ComponentModel.DoWorkEventHandler(this.bgWorker_DoWork);
            this.bgWorker.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.bgWorker_RunWorkerCompleted);
            // 
            // grpAlgorithm
            // 
            this.grpAlgorithm.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.grpAlgorithm.Controls.Add(this.lblAlgorithm);
            this.grpAlgorithm.Controls.Add(this.lblCSVFile);
            this.grpAlgorithm.Controls.Add(this.txtCSVFile);
            this.grpAlgorithm.Controls.Add(this.btnBrowser);
            this.grpAlgorithm.Controls.Add(this.btnRunModel);
            this.grpAlgorithm.Controls.Add(this.cmbAlgorithm);
            this.grpAlgorithm.Location = new System.Drawing.Point(228, 149);
            this.grpAlgorithm.Name = "grpAlgorithm";
            this.grpAlgorithm.Size = new System.Drawing.Size(635, 235);
            this.grpAlgorithm.TabIndex = 7;
            this.grpAlgorithm.TabStop = false;
            this.grpAlgorithm.Text = "Algorithm and Progress";
            // 
            // lblAlgorithm
            // 
            this.lblAlgorithm.AutoSize = true;
            this.lblAlgorithm.Location = new System.Drawing.Point(25, 107);
            this.lblAlgorithm.Name = "lblAlgorithm";
            this.lblAlgorithm.Size = new System.Drawing.Size(50, 13);
            this.lblAlgorithm.TabIndex = 6;
            this.lblAlgorithm.Text = "Algorithm";
            // 
            // lblCSVFile
            // 
            this.lblCSVFile.AutoSize = true;
            this.lblCSVFile.Location = new System.Drawing.Point(15, 75);
            this.lblCSVFile.Name = "lblCSVFile";
            this.lblCSVFile.Size = new System.Drawing.Size(69, 13);
            this.lblCSVFile.TabIndex = 5;
            this.lblCSVFile.Text = "Training CSV";
            // 
            // statusStrip1
            // 
            this.statusStrip1.Location = new System.Drawing.Point(0, 399);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(875, 22);
            this.statusStrip1.TabIndex = 8;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // FormMain
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(875, 421);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.grpAlgorithm);
            this.Controls.Add(this.grpProcessing);
            this.Controls.Add(this.btnLaunch);
            this.Name = "FormMain";
            this.Text = "Free Discovery";
            this.Load += new System.EventHandler(this.FormMain_Load);
            this.grpProcessing.ResumeLayout(false);
            this.grpAlgorithm.ResumeLayout(false);
            this.grpAlgorithm.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnLaunch;
        private System.Windows.Forms.ComboBox cmbAlgorithm;
        private System.Windows.Forms.Button btnBrowser;
        private System.Windows.Forms.TextBox txtCSVFile;
        private System.Windows.Forms.Button btnRunModel;
        private System.Windows.Forms.ProgressBar progressBar1;
        private System.Windows.Forms.GroupBox grpProcessing;
        private System.ComponentModel.BackgroundWorker bgWorker;
        private System.Windows.Forms.Label lblProcess;
        private System.Windows.Forms.GroupBox grpAlgorithm;
        private System.Windows.Forms.Label lblCSVFile;
        private System.Windows.Forms.Label lblAlgorithm;
        private System.Windows.Forms.StatusStrip statusStrip1;
    }
}

