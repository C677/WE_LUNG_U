# WE_LUNG_U
using Docker and Flask

- Hyojeong Chang
- Jonggeun Park
- Minjoo Lee

## How to run 
### 1. Start server
/WE_LUNG_U/we_Lung_u_flask/
```
$ docker-compose up
```

### 2. Access localhost
Open your browser and type that address below!
```
localhost
```
or
```
http://0.0.0.0:80/
```
********************************************************************************

**How to solve...**
- When edited your code is not working
1. Remove docker container
``` 
$ docker rm -f $(docker ps -aq)
```

2. Remove docker images
``` 
$ docker rmi -f $(docker images -a -q)
```

3. Then, restart from scratch!
