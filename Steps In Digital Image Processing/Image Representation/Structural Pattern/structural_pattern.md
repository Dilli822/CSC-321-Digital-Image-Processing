#### Shape Features:

##### Compactness: 
###### Formula: Compactness = (Perimeter^2) / (4π * Area)
- Perimeter: The length of the contour (calculated using cv2.arcLength).
- Area: The area enclosed by the contour (calculated using cv2.contourArea).

##### Eccentricity: 
###### Formula: Eccentricity = sqrt(1 - (Minor Axis / Major Axis)^2)
- Major Axis: The longest axis of an ellipse fitted to the contour.
- Minor Axis: The shortest axis of the fitted ellipse.

##### Circularity: 
###### Formula: Circularity = (4π * Area) / (Perimeter^2)
- Circularity measures how closely the shape resembles a perfect circle.


##### Moment Invariants (Hu Moments):

Hu Moments: 
Derived from image moments (spatial and central moments) using complex formulas.
Formula: Hu moments are based on central moments and are invariant to scaling, translation, and rotation.
- Computed using cv2.HuMoments after finding the contour.


##### Texture Features:

Gray Level Co-occurrence Matrix (GLCM):
The GLCM is used to compute texture-related properties. It captures the frequency with which pairs of pixel intensities occur.

##### Contrast: 
###### Formula: Contrast = ΣΣ(i - j)^2 * P(i, j)
- Contrast measures the intensity difference between a pixel and its neighbor.

##### Correlation: 
###### Formula: Correlation = ΣΣ ((i - μ_i)(j - μ_j) P(i, j)) / (σ_i * σ_j)
- Correlation measures the degree to which pixels are linearly related.

##### Homogeneity: 
###### Formula: Homogeneity = ΣΣ P(i, j) / (1 + |i - j|)
- Homogeneity measures how similar a pixel is to its neighbors.

##### Entropy: 
###### Formula: Entropy = -ΣΣ P(i, j) * log2(P(i, j))
- Entropy measures the randomness in the texture.
