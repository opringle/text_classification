#An mechanical engineer's guide to MXNet: Bring a Bazooka to knife fight

Personally I've never really cared that much for inference.  By this I do not mean that statistical inference is not incredibly valuable in business.  I mean that I find it far more exciting to make predictions I do not have to (and cannot) explain.... but that drive crazy impacts industry.

Recently I was talking with one of Tesla's design engineers (by talking I mean desperately pleading for a job). He  mentioned their team were researching their future goal of fitting neural networks to their car sensor data, not to help it drive, but to predict the probability of part failure in real time.  Machine Learning algorithms like this would allow each Tesla owner to have a completely personalized service interval, based on how they drive it.  Sounds cool right?!!

Having recently retrained in Canada as a Data Scientist, I understand a good amount of the theory behind these algorithms, but having only had roughly a year's coding experience I struggle with writing code for deep learning at a high level.  I don't mean putting together a neural network in Keras or scikit-learn. I'm talking about understanding how to design, tune and and deploy neural networks trained with hundreds of GPUs on massive data quantities (maybe a 1Tb of car sensor data for example).

So with that lets both get started learning.  I suggest you pour a large glass of wine.

#Mxnet: bring a bazooka to a knife fight

This is what Amazon's internal data science team use for the above.  I chose it because they did but also [this video](https://www.youtube.com/watch?v=ScRtj2bNMJE) says:

(+) most open source (apache licence)
(+) front end language makes zero difference to performance since code is compile to C++ before being executed.  
(+) This also means you can port between languages easily.  For example deploy your application to a smartphone.
(+) most efficient and therefore cheapest distributed deep learning library to run on AWS
(+) mix imperitive and declarative styles of programming (write code like an idiot or really well with similar results)
(+) highly memory efficent (again this makes it cheaper/faster)

Finally their [opening paragraph](http://gluon.mxnet.io/) from [their documentation](http://gluon.mxnet.io/) pretty much perfectly matched what I'm looking for.

## The basics

We want the best/easiest operating system to run this software on. That's the latest ubuntu.

The easiest way to install MXNet is to [follow MXnet's 5 step guide](https://mxnet.incubator.apache.org/get_started/install.html).  Select your own computer's OS followed by `python`, `gpu` and `docker` options if you want to follow along.  You'll have to create a free NVIDIA account.  Also, there are a bazillion [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) options.  Just choose the latest version for your OS.

Problem #1.... My mac has an AMD gpu, not an NVIDEA one. Even with a docker container I can't execute the code locally.  I guess that explains this:

![](../images/nvidia_stock_price.png)

Alright so we have to do this on the cloud, which means we have to pay every single time we want to train a model for fun, not for a business. Luckily, I work for a business that would find this shit really useful so lets carry on.

Those of you with an NVIDIA gpu feel free to smugly continue your install and follow along from there.  Everyone else lets go to [the install guide](https://mxnet.incubator.apache.org/get_started/install.html) and use the Amazon CLOUD-GPU-UBUNTU image (you'll need to sign up for an AWS account also which is a pain in the arse....).  We need GPU's for what we plan to do later.

## AWS signup

Install AWS command line tools which on mac was: `$ sudo pip install awscli --ignore-installed six` 

add autocompletion of aws cli commands: `$ complete -C aws_completer aws`

configure your account with `$ aws configure`, [set up a user with permissions copy your security settings into the configure steps](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)

## Connecting to your cloud instance

Select the latest version, `t2.micro` image (which costs $0.01 per hour), make it available by ssh only from yourIP. Click launch.

Go to your EC2 managment console, under the instances tab select connect with your newly launched image highlighted.  Follow the instructions to ssh into the instance.  You should now be in your terminal, ready to start the tutorial.

This is now costing you 0.01 per hour.  If you ever want to pause the image and stop paying simply type `$ aws ec2 stop-instances --instance-ids <your instance ID>` in your local terminal.  Start it again with `$ aws ec2 start-instances --instance-ids <your instance ID>`.  Stopping and starting charges you for 1 minute of usage (0.01/60 dollars).

## Making your cloud instance nice to work with: SFTP from your text editor

I used VSCode for editing.  Since you are following along this may not be so crucial.  However, to set up remote file editing in VSCode follow these steps:

- in VSCode install Remote VSCode, then cmd + shft + p and search user settings.
- add the following fields to the json:

```
"remote.port" : 52698,
"remote.onstartup": true,
```

Now you can connect yoour ec2 instance from the terminal in your editor to keep your ec2 instance in one window.  This time use `ssh -R 52698:localhost:52698 <Public DNS>`

Now we need to install rmate for remote file editing.

```
mkdir ~/bin
curl -Lo ~/bin/rmate https://raw.github.com/textmate/rmate/master/bin/rmate
chmod a+x ~/bin/rmate
export PATH="$PATH:$HOME/bin"
```

Now in your instance terminal you can type `rmate <filename>` and it will open in your editor!

Personally that's enough for the night for me so lets logout by typing `$ exit` and then stop the instance in our local terminal using `$ stop-instances --instance-ids <your instance ID>`

## The fun part

Okay now we are ready to follow this [guide](http://gluon.mxnet.io/).













