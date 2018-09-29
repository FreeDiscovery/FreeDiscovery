using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FreeDiscoveryApp
{
    public class Algorithm
    {
        private int code;
        private string name;
        
        public int Code
        {
            get { return this.code; }
            set { this.code = value; }
        }

        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

        public Algorithm()
        {
            this.code = 0;
            this.name = string.Empty;
        }
    }
}
