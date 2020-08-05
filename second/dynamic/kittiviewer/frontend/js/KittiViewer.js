// Hardcoded paths.
CONFIG_PATH = '/home/sancha/repos/second.pytorch/second/configs/synthia/second/base.yaml';
CHECKPOINT_PATH = "/home/sancha/repos/second.pytorch/second/sid_trained_models/interpDP/synthia/second/base/voxelnet-15775.tckpt";

var KittiViewer = function (pointCloud, lcCloud, lcNetInput, sbCloud, logger, imagePanel, imageCanvas, bevHmapManager) {
    // this.rootPath = "/home/sancha/data/kitti_detection";
    // this.infoPath = "/home/sancha/data/kitti_detection/kitti_infos_train.pkl";
    this.detPath = "/path/to/results.pkl";
    this.backend = "http://127.0.0.1:16666";
    this.checkpointPath = CHECKPOINT_PATH;
    this.configPath = CONFIG_PATH;
    this.drawDet = false;
    this.imageIndex = 0;
    this.gtBoxes = [];
    this.dtBoxes = [];
    this.gtBboxes = [];
    this.dtBboxes = [];
    this.pointCloud = pointCloud;
    this.lcCloud = lcCloud;
    this.lcNetInput = lcNetInput;
    this.sbCloud = sbCloud; 
    this.heatmapPlane = null;
    this.maxPoints = 500000;
    this.spriteColor = new THREE.Color(0x9099ba);
    this.pointVertices = new Float32Array(this.maxPoints * 3);
    this.gtBoxColor = "#00ff00";
    this.dtBoxColor = "#ff0000";
    this.gtLabelColor = "#7fff00";
    this.dtLabelColor = "#ff7f00";
    this.logger = logger;
    this.imagePanel = imagePanel;
    this.imageCanvas = imageCanvas;
    this.image = '';
    this.enableInt16 = true;
    this.int16Factor = 100;
    this.removeOutside = false;
    this.bevHmapManager = bevHmapManager;

    this.connectStreams();
};

KittiViewer.prototype = {
    readCookies : function(){
        if (CookiesKitti.get("kittiviewer_dataset_cname")){
            this.datasetClassName = CookiesKitti.get("kittiviewer_dataset_cname");
        }
        if (CookiesKitti.get("kittiviewer_backend")){
            this.backend = CookiesKitti.get("kittiviewer_backend");
        }
        // if (CookiesKitti.get("kittiviewer_rootPath")){
        //     this.rootPath = CookiesKitti.get("kittiviewer_rootPath");
        // }
        if (CookiesKitti.get("kittiviewer_detPath")){
            this.detPath = CookiesKitti.get("kittiviewer_detPath");
        }
        if (CookiesKitti.get("kittiviewer_checkpointPath")){
            this.checkpointPath = CookiesKitti.get("kittiviewer_checkpointPath");
        }
        // if (CookiesKitti.get("kittiviewer_configPath")){
        //     this.configPath = CookiesKitti.get("kittiviewer_configPath");
        // }
        // if (CookiesKitti.get("kittiviewer_infoPath")){
        //     this.infoPath = CookiesKitti.get("kittiviewer_infoPath");
        // }
    },
    addhttp: function (url) {
        if (!/^https?:\/\//i.test(url)) {
            url = 'http://' + url;
        }
        return url
    },
    buildNet: function( ){
        let self = this;
        let data = {};
        data["checkpoint_path"] = this.checkpointPath;
        data["config_path"] = this.configPath;
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/build_network',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("build kitti det fail!");
                console.log("build kitti det fail!");
            },
            success: function (response) {
                self.logger.message("build kitti det success!");
            }
        });
    },
    toggleGtBoxes: function( ){
        let self = this;
        if (self.gtBoxes.length == 0) {
            return;
        }
        if (self.gtBoxes[0].parent == scene) {
            for (var i = 0; i < self.gtBoxes.length; ++i) {
                scene.remove(self.gtBoxes[i]);
                // Hack: remove and add labels.
                for (var j = 0; j < self.gtBoxes[i].children.length; ++j) {
                    var label = self.gtBoxes[i].children[j];
                    self.gtBoxes[i].remove(label);
                    self.gtBoxes[i].add(label);
                }
            }
        } else {
            for (var i = 0; i < self.gtBoxes.length; ++i) {
                scene.add(self.gtBoxes[i]);
            }
        }
    },
    toggleDtBoxes: function( ){
        let self = this;
        if (self.dtBoxes.length == 0) {
            return;
        }
        if (self.dtBoxes[0].parent == scene) {
            for (var i = 0; i < self.dtBoxes.length; ++i) {
                scene.remove(self.dtBoxes[i]);
                // Hack: remove and add labels.
                for (var j = 0; j < self.dtBoxes[i].children.length; ++j) {
                    var label = self.dtBoxes[i].children[j];
                    self.dtBoxes[i].remove(label);
                    self.dtBoxes[i].add(label);
                }
            }
        } else {
            for (var i = 0; i < self.dtBoxes.length; ++i) {
                scene.add(self.dtBoxes[i]);
            }
        }
    },
    toggleSceneCloud: function( ){
        let self = this;
        self.pointCloud.visible = !self.pointCloud.visible;
    },
    toggleSBL: function( ){
        let self = this;
        self.sbCloud.visible = !self.sbCloud.visible;
    },
    toggleLC: function( ){
        let self = this;

        // 1st mode: lcCloud.
        if (self.lcCloud.visible && !self.lcNetInput.visible) {
            self.lcCloud.visible = false;
            self.lcNetInput.visible = true;
        }
        // 2nd mode: lcNetInput visible.
        else if (!self.lcCloud.visible && self.lcNetInput.visible) {
            self.lcCloud.visible = false;
            self.lcNetInput.visible = false;
        }
        // 3rd mode: both invisible.
        else {
            self.lcCloud.visible = true;
            self.lcNetInput.visible = false;
        }
    },
    makeEccvVisualization: async function( ){
        let self = this;

        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        async function save_fig(fname) {
            await sleep(1000);
            animate();
            var imgData;
            fname = `${fname}.jpg`
            try {
                var strMime = "image/jpeg";
                var strDownloadMime = "image/octet-stream";
                imgData = renderer.domElement.toDataURL(strMime);
                self.saveFile(imgData.replace(strMime, strDownloadMime), fname);
            } catch (e) {
                console.log(e);
                return;
            }
        }
        
        // * SCENE.
        self.toggleSBL();  // off
        bloomPass.strength = 0;
        pointParicle.material.size = 10;
        await save_fig('scene')

        // Transition to images without scene cloud.
        self.toggleSceneCloud() // off
        bloomPass.strength = 2;
        pointParicle.material.size = 10;
        self.toggleGtBoxes();  // off

        // * SBL hm
        self.inference_next();
        await sleep(5000);

        self.toggleSBL();  // on
        self.toggleHeatmap();  // entropy
        self.toggleDtBoxes();  // off
        await save_fig('0_hm_sb');

        // * SBL
        self.toggleHeatmap();  // off
        self.toggleDtBoxes();  // on
        self.toggleLC();  // lcNetInput
        await save_fig('0_dt_in')

        // Switch to light curtains.
        // SBL    : on
        // Heatmap: off.
        // DtBoxes: on.
        // LC mode: lcNetInput.

        for (i = 1; i <= 10; i+=1) {
            self.inference_next();
            await sleep(5000);

            self.toggleSBL();  // off
            self.toggleHeatmap(); self.toggleHeatmap(); // entropy
            self.toggleDtBoxes(); // off
            self.toggleLC(); self.toggleLC(); // lcCloud
            await save_fig(`${i}_hm_lc`);

            self.toggleSBL(); // on
            self.toggleHeatmap(); // off
            self.toggleDtBoxes(); // on
            self.toggleLC();  // lcNetInput
            await save_fig(`${i}_dt_in`);
        }

        // // * SCENE.
        // bloomPass.strength = 0;
        // pointParicle.material.size = 10;
        // self.toggleSceneCloud();  // on
        // self.toggleGtBoxes();  // on
        // self.toggleDtBoxes();  // off
        // self.toggleLC();  // off
        // await save_fig('scene')
    },
    _inference: function(response) {
        let self = this;

        response = response["results"][0];
        if (! ("lc_cloud") in response) {
            return;
        }

        var locs = response["dt_locs"];
        var dims = response["dt_dims"];
        var rots = response["dt_rots"];
        var scores = response["dt_scores"];
        self.dtBboxes = response["dt_bbox"];
        for (var i = 0; i < self.dtBoxes.length; ++i) {
            for (var j = self.dtBoxes[i].children.length - 1; j >= 0; j--) {
                self.dtBoxes[i].remove(self.dtBoxes[i].children[j]);
            }
            scene.remove(self.dtBoxes[i]);
            self.dtBoxes[i].geometry.dispose();
            self.dtBoxes[i].material.dispose();
        }
        let label_with_score = [];
        for (var i = 0; i < locs.length; ++i) {
            label_with_score.push(scores[i].toFixed(2).toString());
        }
        
        self.dtBoxes = boxEdgeWithLabel(dims, locs, rots, 2, self.dtBoxColor,
            label_with_score, self.dtLabelColor);
        for (var i = 0; i < self.dtBoxes.length; ++i) {
            scene.add(self.dtBoxes[i]);
        }
        // SID: No need to re-draw the image when the "next" of "prev" button is clicked.
        // self.drawImage();

        // Add/update heatmap.
        self.bevHmapManager.updateHmap(response["confidenceMap_b64"], response["entropyMap_b64"]);
        // if (self.heatmapPlane != null) {
        //     scene.remove(self.heatmapPlane);
        //     self.heatmapPlane.geometry.dispose();
        //     self.heatmapPlane.material.dispose();
        //     self.heatmapPlane = null;
        // }

        // Add LC point cloud (visible).
        var points_buf = str2buffer(atob(response["lc_cloud"]));
        var points;
        if (self.enableInt16){
            var points = new Int16Array(points_buf);
        }
        else{
            var points = new Float32Array(points_buf);
        }
        for (var i = 0; i < Math.min(points.length / 4, self.maxPoints); i++) {
            var x = points[4 * i];
            var y = points[4 * i + 1];
            var z = points[4 * i + 2];
            var intensity = points[4 * i + 3];
            
            if (self.enableInt16) {
                x /= self.int16Factor;
                y /= self.int16Factor;
                z /= self.int16Factor;
                intensity /= self.int16Factor;
            }

            // Position
            self.lcCloud.geometry.attributes.position.array[i * 3] = x;
            self.lcCloud.geometry.attributes.position.array[i * 3 + 1] = y;
            self.lcCloud.geometry.attributes.position.array[i * 3 + 2] = z;

            // Color.
            self.lcCloud.geometry.attributes.color.array[3 * i] = 0;
            self.lcCloud.geometry.attributes.color.array[3 * i + 1] = intensity;
            self.lcCloud.geometry.attributes.color.array[3 * i + 2] = 0.2;
        }
        self.lcCloud.geometry.setDrawRange(0, Math.min(points.length / 4,
            self.maxPoints));
        self.lcCloud.geometry.attributes.position.needsUpdate = true;
        self.lcCloud.geometry.attributes.color.needsUpdate = true;
        self.lcCloud.geometry.computeBoundingSphere();

        // Add LC net input cloud (invisible).
        var points_buf = str2buffer(atob(response["lc_net_input"]));
        var points;
        if (self.enableInt16){
            var points = new Int16Array(points_buf);
        }
        else{
            var points = new Float32Array(points_buf);
        }
        for (var i = 0; i < Math.min(points.length / 4, self.maxPoints); i++) {
            var x = points[4 * i];
            var y = points[4 * i + 1];
            var z = points[4 * i + 2];
            var intensity = points[4 * i + 3];
            
            if (self.enableInt16) {
                x /= self.int16Factor;
                y /= self.int16Factor;
                z /= self.int16Factor;
                intensity /= self.int16Factor;
            }

            // Position
            self.lcNetInput.geometry.attributes.position.array[i * 3] = x;
            self.lcNetInput.geometry.attributes.position.array[i * 3 + 1] = y;
            self.lcNetInput.geometry.attributes.position.array[i * 3 + 2] = z;

            // Color.
            self.lcNetInput.geometry.attributes.color.array[3 * i] = 0;
            self.lcNetInput.geometry.attributes.color.array[3 * i + 1] = intensity;
            self.lcNetInput.geometry.attributes.color.array[3 * i + 2] = 0.2;
        }
        self.lcNetInput.geometry.setDrawRange(0, Math.min(points.length / 4,
            self.maxPoints));
        self.lcNetInput.geometry.attributes.position.needsUpdate = true;
        self.lcNetInput.geometry.attributes.color.needsUpdate = true;
        self.lcNetInput.geometry.computeBoundingSphere();
    },
    inference_next: function( ){
        let self = this;
        let data = {"image_idx": self.imageIndex, "remove_outside": self.removeOutside};
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/inference_next_by_idx',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("inference fail!");
                console.log("inference fail!");
            },
            success: function (response) {
                self._inference(response)
            }
        });
    },
    inference_prev: function( ){
        let self = this;
        let data = {"image_idx": self.imageIndex, "remove_outside": self.removeOutside};
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/inference_prev_by_idx',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("inference fail!");
                console.log("inference fail!");
            },
            success: function (response) {
                self._inference(response)
            }
        });
    },
    plot: function () {
        return this._plot(this.imageIndex);
    },
    next: function () {
        this.imageIndex += 1;
        return this.plot();
    },
    prev: function () {
        if (this.imageIndex > 0) {
            this.imageIndex -= 1;
            return this.plot();
        }
    },
    clear: function(){
        for (var i = 0; i < this.gtBoxes.length; ++i) {
            for (var j = this.gtBoxes[i].children.length - 1; j >= 0; j--) {
                this.gtBoxes[i].remove(this.gtBoxes[i].children[j]);
            }
            scene.remove(this.gtBoxes[i]);
            this.gtBoxes[i].geometry.dispose();
            this.gtBoxes[i].material.dispose();
        }
        this.gtBoxes = [];
        for (var i = 0; i < this.dtBoxes.length; ++i) {
            for (var j = this.dtBoxes[i].children.length - 1; j >= 0; j--) {
                this.dtBoxes[i].remove(this.dtBoxes[i].children[j]);
            }
            scene.remove(this.dtBoxes[i]);
            this.dtBoxes[i].geometry.dispose();
            this.dtBoxes[i].material.dispose();
        }
        this.dtBoxes = [];
        this.gtBboxes = [];
        this.dtBboxes = [];
        // this.image = '';
    },
    connectSceneCloudStream: function() {
        let self = this;
        var sceneCloudSource = new EventSource(self.addhttp(self.backend) + "/api/stream_scene_cloud");
        sceneCloudSource.onmessage = function(e) {
            var points_buf = str2buffer(atob(e.data));
            var points;
            if (self.enableInt16){
                var points = new Int16Array(points_buf);
            }
            else{
                var points = new Float32Array(points_buf);
            }
            
            var numFeatures = 6;
            
            // Set positions.
            for (var i = 0; i < Math.min(points.length / numFeatures, self.maxPoints); i++) {
                let x = points[i * numFeatures + 0];
                let y = points[i * numFeatures + 1];
                let z = points[i * numFeatures + 2];
                if (self.enableInt16) {
                    x /= self.int16Factor;
                    y /= self.int16Factor;
                    z /= self.int16Factor
                }
                self.pointCloud.geometry.attributes.position.array[i * 3 + 0] = x;
                self.pointCloud.geometry.attributes.position.array[i * 3 + 1] = y;
                self.pointCloud.geometry.attributes.position.array[i * 3 + 2] = z;
            }

            // Set color.
            for (var i = 0; i < Math.min(points.length / numFeatures, self.maxPoints); i++) {
                if (numFeatures == 6) {
                    r = points[i * numFeatures + 3] / 255;
                    g = points[i * numFeatures + 4] / 255;
                    b = points[i * numFeatures + 5] / 255;

                    if (self.enableInt16) {
                        r /= self.int16Factor;
                        b /= self.int16Factor;
                        g /= self.int16Factor
                    }
                } else {
                    r = self.spriteColor.r;
                    g = self.spriteColor.g;
                    b = self.spriteColor.b;
                }
                self.pointCloud.geometry.attributes.color.array[i * 3 + 0] = r;
                self.pointCloud.geometry.attributes.color.array[i * 3 + 1] = g;
                self.pointCloud.geometry.attributes.color.array[i * 3 + 2] = b;
            }
                
            self.pointCloud.geometry.setDrawRange(0, Math.min(points.length / numFeatures,
                self.maxPoints));
            self.pointCloud.geometry.attributes.position.needsUpdate = true;
            self.pointCloud.geometry.attributes.color.needsUpdate = true;
            self.pointCloud.geometry.computeBoundingSphere();
        };
        console.log("Connected to scene cloud stream.")
    },
    connectCameraImageStream: function() {
        let self = this;
        var cameraImageSource = new EventSource(self.addhttp(self.backend) + "/api/stream_camera_image");
        cameraImageSource.onmessage = function(e) {
            this.image = e.data;
            var image = new Image();
            image.onload = function() {
                let aspect = image.width / image.height;
                let w = self.imageCanvas.width;
                self.imageCanvas.height = w / aspect;
                let h = self.imageCanvas.height;
                let ctx = self.imageCanvas.getContext("2d");
                ctx.drawImage(image, 0, 0, w, h);
            };
            image.src = this.image;
        };
        console.log("Connected to camera image stream.")
    },
    connectSbCloudStream: function() {
        let self = this;
        var sbCloudSource = new EventSource(self.addhttp(self.backend) + "/api/stream_lidar_cloud");
        sbCloudSource.onmessage = function(e) {
            var sb_points_buf = str2buffer(atob(e.data));
            var sb_points;
            if (self.enableInt16){
                var sb_points = new Int16Array(sb_points_buf);
            }
            else{
                var sb_points = new Float32Array(sb_points_buf);
            }
            // Set positions.
            for (var i = 0; i < Math.min(sb_points.length / 3, self.maxPoints); i++) {
                let x = sb_points[i * 3 + 0];
                let y = sb_points[i * 3 + 1];
                let z = sb_points[i * 3 + 2];
                if (self.enableInt16) {
                    x /= self.int16Factor;
                    y /= self.int16Factor;
                    z /= self.int16Factor
                }
                self.sbCloud.geometry.attributes.position.array[i * 3 + 0] = x;
                self.sbCloud.geometry.attributes.position.array[i * 3 + 1] = y;
                self.sbCloud.geometry.attributes.position.array[i * 3 + 2] = z;
            }
            self.sbCloud.geometry.setDrawRange(0, Math.min(sb_points.length / 3, self.maxPoints));
            self.sbCloud.geometry.attributes.position.needsUpdate = true;
            self.sbCloud.geometry.computeBoundingSphere();
        };
        console.log("Connected to lidar cloud stream.")
    },
    connectLcCloudStream: function() {
        let self = this;
        var lcCloudSource = new EventSource(self.addhttp(self.backend) + "/api/stream_lc_cloud");
        lcCloudSource.onmessage = function(e) {
            // Add LC point cloud (visible).
            var points_buf = str2buffer(atob(e.data));
            var points;
            if (self.enableInt16){
                var points = new Int16Array(points_buf);
            }
            else{
                var points = new Float32Array(points_buf);
            }
            for (var i = 0; i < Math.min(points.length / 4, self.maxPoints); i++) {
                var x = points[4 * i];
                var y = points[4 * i + 1];
                var z = points[4 * i + 2];
                var intensity = points[4 * i + 3];
                
                if (self.enableInt16) {
                    x /= self.int16Factor;
                    y /= self.int16Factor;
                    z /= self.int16Factor;
                    intensity /= self.int16Factor;
                }

                // Position
                self.lcCloud.geometry.attributes.position.array[i * 3] = x;
                self.lcCloud.geometry.attributes.position.array[i * 3 + 1] = y;
                self.lcCloud.geometry.attributes.position.array[i * 3 + 2] = z;

                // Color.
                self.lcCloud.geometry.attributes.color.array[3 * i] = 0;
                self.lcCloud.geometry.attributes.color.array[3 * i + 1] = intensity;
                self.lcCloud.geometry.attributes.color.array[3 * i + 2] = 0.2;
            }
            self.lcCloud.geometry.setDrawRange(0, Math.min(points.length / 4,
                self.maxPoints));
            self.lcCloud.geometry.attributes.position.needsUpdate = true;
            self.lcCloud.geometry.attributes.color.needsUpdate = true;
            self.lcCloud.geometry.computeBoundingSphere();
        };
        console.log("Connected to light curtain cloud stream.")
    },
    connectDtBoxesStream: function() {
        let self = this;
        var dtBoxesSource = new EventSource(self.addhttp(self.backend) + "/api/stream_dt_boxes");
        dtBoxesSource.onmessage = function(e) {
            // Add LC point cloud (visible).
            var detections = JSON.parse(e.data);
            var locs = detections["dt_locs"];
            var dims = detections["dt_dims"];
            var rots = detections["dt_rots"];
            var scores = detections["dt_scores"];
            self.dtBboxes = detections["dt_bbox"];
            for (var i = 0; i < self.dtBoxes.length; ++i) {
                for (var j = self.dtBoxes[i].children.length - 1; j >= 0; j--) {
                    self.dtBoxes[i].remove(self.dtBoxes[i].children[j]);
                }
                scene.remove(self.dtBoxes[i]);
                self.dtBoxes[i].geometry.dispose();
                self.dtBoxes[i].material.dispose();
            }
            let label_with_score = [];
            for (var i = 0; i < locs.length; ++i) {
                label_with_score.push(scores[i].toFixed(2).toString());
            }
            
            self.dtBoxes = boxEdgeWithLabel(dims, locs, rots, 2, self.dtBoxColor,
                label_with_score, self.dtLabelColor);
            for (var i = 0; i < self.dtBoxes.length; ++i) {
                scene.add(self.dtBoxes[i]);
            }
        };
        console.log("Connected to dt boxes stream.")
    },
    connectEntropyMapStream: function() {
        let self = this;
        var entropyMapSource = new EventSource(self.addhttp(self.backend) + "/api/stream_entropy_map");
        entropyMapSource.onmessage = function(e) {
            self.bevHmapManager.updateHmap(null, e.data);
        };
        console.log("Connected to entropy map stream.")
    },
    connectStreams: function() {
        let self = this;
        self.connectSceneCloudStream();
        // self.connectCameraImageStream();
        // self.connectSbCloudStream();
        self.connectLcCloudStream();
        // self.connectDtBoxesStream();
        // self.connectEntropyMapStream();
    },
    _plot: function (image_idx) {
        let data = {};
        data["video_idx"] = image_idx;
        data["with_det"] = this.drawDet;
        data["enable_int16"] = this.enableInt16;
        data["int16_factor"] = this.int16Factor;
        data["remove_outside"] = this.removeOutside;
        let self = this;
        var ajax1 = $.ajax({
            url: this.addhttp(this.backend) + '/api/run_simulation',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("run_simulation fail!!");
                console.log("run_simulation fail!!");
            },
            success: function (response) {
                console.log("run_simulation request completed!");
            }
        });
    },
    stopSimulation: function () {
        let data = {};
        let self = this;
        console.log("Sending stop_simulation request ...");
        var ajax1 = $.ajax({
            url: this.addhttp(this.backend) + '/api/stop_simulation',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            success: function (response) {
                console.log("stop_simulation request completed!");
            }
        });
    },    
    drawImage : function(){
        if (this.image === ''){
            console.log("??????");
            return;
        }
        let self = this;
        var image = new Image();
        // image.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNk+M9Qz0AEYBxVSF+FAAhKDveksOjmAAAAAElFTkSuQmCC";
        // console.log(response["image_b64"]);
        image.onload = function() {
            let aspect = image.width / image.height;
            let w = self.imageCanvas.width;
            self.imageCanvas.height = w / aspect;
            let h = self.imageCanvas.height;
            let ctx = self.imageCanvas.getContext("2d");
            console.log("draw image");
            ctx.drawImage(image, 0, 0, w, h);
            let x1, y1, x2, y2;
            /*
            for (var i = 0; i < self.gtBboxes.length; ++i){
                ctx.beginPath();
                x1 = self.gtBboxes[i][0] * w;
                y1 = self.gtBboxes[i][1] * h;
                x2 = self.gtBboxes[i][2] * w;
                y2 = self.gtBboxes[i][3] * h;
                ctx.rect(x1, y1, x2 - x1, y2 - y1);
                ctx.lineWidth = 1;
                ctx.strokeStyle = "green";
                ctx.stroke();    
            }
            for (var i = 0; i < self.dtBboxes.length; ++i){
                ctx.beginPath();
                x1 = self.dtBboxes[i][0] * w;
                y1 = self.dtBboxes[i][1] * h;
                x2 = self.dtBboxes[i][2] * w;
                y2 = self.dtBboxes[i][3] * h;
                ctx.rect(x1, y1, x2 - x1, y2 - y1);
                ctx.lineWidth = 1;
                ctx.strokeStyle = "blue";
                ctx.stroke();    
            }
            */
        };
        image.src = this.image;

    },
    saveAsImage: function(renderer) {
        var imgData, imgNode;
        try {
            var strMime = "image/jpeg";
            var strDownloadMime = "image/octet-stream";
            imgData = renderer.domElement.toDataURL(strMime);
            this.saveFile(imgData.replace(strMime, strDownloadMime), `pc_${this.imageIndex}.jpg`);
        } catch (e) {
            console.log(e);
            return;
        }
    },
    saveFile : function (strData, filename) {
        var link = document.createElement('a');
        if (typeof link.download === 'string') {
            document.body.appendChild(link); //Firefox requires the link to be in the body
            link.download = filename;
            link.href = strData;
            link.click();
            document.body.removeChild(link); //remove the link when done
        } else {
            location.replace(uri);
        }
    }

}

var BevHmapManager = function(bevPanel, bevCanvas) {
    // Add hmapPlane: initialized with no texture and invisible.
    this.hmapPlane = createHeatmapPlane();  // with dummy texture
    this.hmapPlane.visible = false;
    scene.add(this.hmapPlane);

    this.bevPanel = bevPanel;
    this.bevCanvas = bevCanvas;
    this.hmapMode = "Entropy";  // either "Confidence" or "Entropy"
    
    // These are null if no heatmap to display.
    // Treat this as a mode -- if null, display orthographic view, else display heatmap.
    this.confidenceMap_b64 = null;
    this.entropyMap_b64 = null;

    // Set up BEV camera
    this.camerabev = new THREE.OrthographicCamera(80, -80, -50, 50, 1, 500);
    this.camerabev.up.set(-1, 0, 1);
    this.camerabev.position.set(0, 0, 50);
    this.camerabev.lookAt(0, 0, 0);

    // Set up BEV renderer
    this.rendererBev = new THREE.WebGLRenderer({
        antialias: true
    });
    // this.rendererBev.setPixelRatio(window.devicePixelRatio);  // uncommenting this causes weird edges
    // this.rendererBev.setSize(wBev, hBev);  // wBev and hBev will be retrieved in drawBevCanvas.
    this.drawBevCanvas();
};

BevHmapManager.prototype = {

    toggleHmapPlaneVisibility : function() {
        let self = this;
        if (self.hmapPlane.visible) {
            // Going from visible to invisible.
            self.hmapPlane.visible = false;
        } else if (self.confidenceMap_b64 != null) {
            // Do not make it visible when there is no heatmap.
            // Going from invisible to visible.
            // Load texture.
            self.hmapPlane.visible = true;
            self.loadHmapTexture();
        }
    },

    toggleHmapMode : function() {
        let self = this;

        // First just update the self.hmapMode variable.
        if (self.hmapMode == "Confidence") self.hmapMode = "Entropy";
        else self.hmapMode = "Confidence";

        // If heatmaps actually exist, draw them on canvas and load them on hmapPlane.
        if (self.entropyMap_b64 != null) {
            self.drawBevCanvas();
            if (self.hmapPlane.visible)
                self.loadHmapTexture();
        }
    },

    loadHmapTexture : function() {
        let self = this;

        // Before calling this function, check if the heatmap (self.confidenceMap_b64) exists.
        if (self.entropyMap_b64 == null)
            throw "SID: cannot call loadHmapTexture() when self.entropyMap_b64 is null.";

        // Dispose current texture.
        self.hmapPlane.material.map.dispose();

        // Load new texture.
        if (self.hmapMode == "Confidence") {
            self.hmapPlane.material.map = new THREE.TextureLoader().load(self.confidenceMap_b64);
        }
        else if (self.hmapMode == "Entropy") {
            self.hmapPlane.material.map = new THREE.TextureLoader().load(self.entropyMap_b64);
        }
    },
    
    drawBevCanvas : function(){
        /*
        Use this function whenver there is an update in what to show on the BEV Canvas.
        This function is never called repeatedly and is not a part of animate(), only called
        when the status changes.
        */
        let self = this;

        // Orthographic projection camera.
        if (self.entropyMap_b64 == null) {
            let w = self.bevCanvas.width;
            let h = self.bevCanvas.height;
            // TODO: check if renderer size has changed.
            self.rendererBev.setSize(w, h);
            let aspect = w / h;
            self.camerabev.left = 0.5 * aspect * 100;
            self.camerabev.right = -0.5 * aspect * 100;
            self.camerabev.top = -0.5 * 100;
            self.camerabev.bottom = 0.5 * 100;
            self.camerabev.updateProjectionMatrix();
            self.bevPanel.headertitle.textContent = "Bird's eye view";
        } else {
            // Draw heatmap on BEV canvas.
            var image = new Image();
            image.onload = function() {
                let aspect = image.height / image.width;  // since heatmap is rotated
                let w = self.bevCanvas.width;
                let h = w / aspect;  // target height
                if (self.bevCanvas.height != h) self.bevCanvas.height = h;
                let ctx = self.bevCanvas.getContext("2d");

                ctx.rotate(-Math.PI / 2);
                ctx.drawImage(image, -h, 0, h, w);
            };

            if (self.hmapMode == 'Confidence') {
                image.src = this.confidenceMap_b64;
                self.bevPanel.headertitle.textContent = "Confidence Map";
            } else if (self.hmapMode == 'Entropy') {
                image.src = this.entropyMap_b64;
                self.bevPanel.headertitle.textContent = "Entropy Map";
            }
        }
    },

    render : function() {
        /* Renders orthographic camera and draws on canvas ONLY IF no heatmap*/
        let self = this;

        if (self.confidenceMap_b64 == null) {
            self.camerabev.updateProjectionMatrix();
            self.rendererBev.render(scene, this.camerabev);
            // composerbev.render(0.01);
            let ctx = self.bevCanvas.getContext("2d");
            ctx.drawImage(self.rendererBev.domElement, 0, 0);
        }
    },

    updateHmap : function(confidenceMap_b64, entropyMap_b64) {
        let self = this;
        
        self.confidenceMap_b64 = confidenceMap_b64;
        self.entropyMap_b64 = entropyMap_b64;

        // Update hmapPlane.
        if (self.hmapPlane.visible) self.loadHmapTexture();

        // Update BEV canvas.
        self.drawBevCanvas();
    },

    removeHmap : function() {
        let self = this;

        self.confidenceMap_b64 = null;
        self.entropyMap_b64 = null;
        self.hmapPlane.visible = false;
        self.drawBevCanvas();
    },
}