from torch.utils.data import Dataset



'''
DATASET CLASS FOR SAM



We test different options for the prompt encoder in the SAM pipeline.

    ### Prompt Encoder: 
    SAM has an prompt encoder part that takes either a list of point queries, a list of bounding boxes or masks as input. How can we use the prompt encoder in our pipeline?

    #### Training

    Use as input either: 
    - No input
    - List of point queries: 
        - Uniformly overlay the whole image with a grid of points
        - Sample coordinates of the street pixels as point queries
    - Bounding box of whole image, part where streets are located, could split up streets into multiple bounding boxes
    - Masks of the street pixels, could leak ground truth information? 

    #### Evaluation/Testing 

    - List of point queries: 
        - Uniformly overlay the whole image with a grid of points
    - No input
    - Bounding box of whole image

'''

class SAM_DataSet(Dataset):

    def __init__(self, satellite_images, street_masks, prompt_options, transform=None):
        self.satellite_images = satellite_images
        self.street_masks = street_masks

        # check if the number of images and masks is the same
        assert len(self.satellite_images) == len(self.street_masks), "Number of images and masks should be the same"

        self.prompt_options = prompt_options

        if self.prompt_options == "None":
            # for each point query, bounding box and mask, pass None            
            self.prompts = [[None, None, None] for _ in range(len(self.satellite_images))]

        elif self.prompt_options == "point_queries_grid":
            # generate point queries
            raise NotImplementedError()
            
        elif self.prompt_options == "bounding_boxes":
            # generate bounding boxes
            raise NotImplementedError()

        elif self.prompt_options == "point_queries_sample_mask":
            # sample from street-masks as point queries
            raise NotImplementedError()

    
        # what about transforms, might want to enable a series of transform here, e.g. rotation, flipping, etc. ...     
        self.transform = transform



    def __len__(self):
        return len(self.satellite_images)

    def __getitem__(self, idx):
        
        sample = self.satellite_images[idx]
        target = self.street_masks[idx]

        prompt = self.prompts[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, prompt, target