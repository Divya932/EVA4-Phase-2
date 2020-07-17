## Deploying DL model as a function to AWS Lambda
Steps followed:  
1. Install docker on your system.  
   Refer the link: https://download.docker.com/linux/ubuntu/dists/bionic/pool/stable/amd64/  
   Download the latest version of these files:  
   a. docker-ce : The Docker Engine package  
   b. docker-cli  
   c. containered.io  
   use sudo dpkg -i containerd.io_1.2.13-2_amd64.deb for example to install all three packages.  
   
   Read more about what is docker and why do we use them here: https://docs.docker.com/get-started/  

2. Install serverless globally  
   npm install serverless -g  
   
3. Login to your AWS serverless account  
   serverless login  
   
4. Create a serverless function  
   serverless create --template Your-template  
   
5. Edit your handler.py and serverless.yml code files generated automatically  

6. Deploy your model on to the cloud provider  
   serverless deploy  
   
7. There goes your deployed function link! (Of a similar type of course!)  
   http://yourfunction.amazonaws.com/your-template  
