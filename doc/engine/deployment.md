# Deployment options


![FreeDiscovery deployment](../_static/FreeDiscovery_infra.png)


## 1. Development server

The options used to start the FreeDiscovery server are defined in `scripts/run_api.py`. By default, the options `processes=1, threaded=True` are used, which allows to run on all platforms, but disables parallel processing in FreeDiscovery.

On Linux, Mac or when running in a Docker container (including on Windows), a more efficient approach is to set `processes=4, threaded=False` (e.g. to run on 4 CPU cores), before starting the server (or before building the Docker container), which would allow parallel hashed feature extraction and cross validations.

## 2. Docker deployment on AWS

This section illustrates how to run a FreeDiscovery Docker images on AWS EC2 without using the EC2 container service, however it can be extended to that purpose. Here we manually create an AMI instance and install docker, but [`docker-machine` with `amazonec2` driver](https://docs.docker.com/machine/drivers/aws/) can also be used to simplify the setup phase.

 1. Choose an AMI (e.g. Amazon Linux AMI) and create an instance with sufficient resources (at least 16GB RAM, 4 CPU cores, twice the RAM size in free disk space, `m4.xlarge` or preferably `c4.2xlarge` to process TREC legal 700,000 document collection)
    * port 5001 is used by default in FreeDiscovery and must be open to incoming connections.
 2. Install Docker in the instance (cf. for instance the ["Docker Basics" AWS EC2 guide](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html#install_docker))
 3. A prebuild Docker image of FreeDiscovery can be downloaded with,
    * [only once] `docker login`  # using your [hub.docker.com](https://hub.docker.com) credentials
    * [only once, optional] requesting permission to access the `freediscovery/freediscovery` image for your `userid`
    * `docker pull freediscovery/freediscovery:<tag>` # where `<tag>` is one of the stable tags on [github.com/FreeDiscovery/FreeDiscovery](http://github.com/FreeDiscovery/FreeDiscovery).

 3. Create or choose a folder where the data to process will be copied and that can be used to store temporary files.
 4. Run Docker and mount the above folder under `/freediscovery_shared` inside the container,
    ```
    docker run -t -i -v /<shared_folder>:/freediscovery_shared -p 5001:5001 freediscovery/freediscovery
    ```

## 3. Starting FreeDiscovery as a daemon

The easiest way of running FreeDiscovery as a background service on Linux / MacOS is by using [supervisor](http://supervisord.org/index.html),

 1. Create a dedicated user account (to avoid running the server as `root`),
    ```
    sudo useradd -r freediscovery
    ```
 2. [Install supvervisor](http://supervisord.org/installing.html#installing-a-distribution-package), for instance
    ```
    sudo apt-get install supervisor
    ```
    on a Debian/Ubuntu based system. Make sure supervisor is started with,
    ```
    sudo service supervisor start
    ```
    
 3. Configure supervisor to launch FreeDiscovery by adding creating a new service definition file `/etc/supervisor/conf.d/freediscovery.conf`
    ```
    [program:freediscovery]
    command=<freediscovery_excutable_path> run --cache-dir /srv/freediscovery_shared/ -y --hostname 0.0.0.0 -p 8888 --log-file /var/log/freediscovery/freediscovery.backend.log
    directory=/
    autostart=true
    autorestart=true
    startretries=1
    redirect_stderr=true
    stdout_logfile=/var/log/freediscovery/freediscovery.stdout.log
    user=freediscovery
    ```
    where `<freediscovery_excutable_path>` is the absolute path to the FreeDiscovery executable (can be found with `which freediscovery`). You should customize the `comand` options as needed (see `freediscovery run --help`). If you are running freediscovery with the `--hostname 0.0.0.0` (i.e. connection open from anywhere) make sure the server has a configured firewall.

 4. Give the `freediscovery` user the necessary permissions,
    ```
    sudo mkdir /srv/freediscovery_shared/
    sudo chown freediscovery /srv/freediscovery_shared/
    sudo mkdir /var/log/freediscovery/
    sudo chown freediscovery /var/log/freediscovery/
    ```

 5. Reload the configuration,
    ```
    sudo supervisorctl reread
    sudo supervisorctl update
    ```
    and check that freediscovery service is running,
    ```
    sudo supervisorctl
    ```
    Going to `http://<server_ip>:<fd_port_number>` in a browser should also yield version information about the running FreeDiscovery server.

 6. The FreeDiscovery service will auto-launch at startup. It can also be manually started/stopped with,
    ```
    sudo supervisorctl [start|stop] freediscovery
    ```
    Please refer to [this post](https://serversforhackers.com/c/monitoring-processes-with-supervisord) for a quick supervisor tutoral, or to [supervisor documentation](http://supervisord.org/index.html) for additional details.





