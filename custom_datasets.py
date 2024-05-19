from torch.utils.data import Dataset
import numpy as np

'''
DATASET CLASS FOR Custom Decoder SAM


'''




class sam_dataset(Dataset):

    def __init__(self, satellite_images, street_masks, prompt_options, transforms=[]):
        
        self.satellite_images = satellite_images
        self.street_masks = street_masks

        # check if the number of images and masks is the same
        assert len(self.satellite_images) == len(self.street_masks), "Number of images and masks should be the same"



        
        self.prompt_options = prompt_options[0]
        self.prompt_detail = prompt_options[1] # contains e.g the number of points for grid generation
        
        if self.prompt_options == "None":
            # for each point query, bounding box and mask, pass None 
            # -> from first testing I think this doenst return any result [mask of only false] using the SAMPredictor .predict()
            self.prompts = [[None, None, None,None]]

        elif self.prompt_options == "point_queries_grid":
            # generate point queries
            num_points_grid_one_axis = self.prompt_detail
            # generate a grid of points 

            step_size = 400 // (num_points_grid_one_axis)

            points = []
            for i in range(num_points_grid_one_axis):
                for j in range(num_points_grid_one_axis):
                    points.append([step_size//2 + i*step_size,step_size//2 + j*step_size])
            points = np.array(points)
            point_labels = np.ones(len(points))
            # TODO try out different point labels....

            # get the number of points for the grid
            self.prompts = [[points, point_labels, None,None]] # if the prompt is identical for all images we just store it once
            
        elif self.prompt_options == "point_queries_sample_mask":
            # sample from street-masks as point queries
            raise NotImplementedError()

        # a list of transformations such as horizontal flip, vertical flip that will be applied to the images and target masks
        self.transforms = transforms

        
    def __len__(self):
        return len(self.satellite_images)

    def __getitem__(self, idx):
        
        sample = self.satellite_images[idx]
        target = self.street_masks[idx]
        
        prompt = self.prompts[0]
        if len(self.prompts) > 1:
            prompt = self.prompts[idx]
    
        for transform in self.transforms:
            # decide whether to apply the transformation to the target or the sample
            random = np.random.rand()
            if random > 0.5:
                target = transform(target)
                sample = transform(sample)
                if prompt is not None:
                    raise NotImplementedError()
                    # TODO: !! WARNING:  depending on the prompts these then also need to be transformed
                    prompt = transform(prompt)
                    
        return (sample, prompt, target)
    




