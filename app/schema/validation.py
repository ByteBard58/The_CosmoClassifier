from pydantic import BaseModel, Field
from typing import Annotated

class UserInput(BaseModel):
  ra : Annotated[float,Field(
    ..., description="Right Ascension (required)",
    ge=0, le=360,examples=[179.632888,183.3014244], allow_inf_nan=False
  )]
  dec : Annotated[float,Field(
    ..., description="Declination (required)",
    ge= -90, le=90, examples=[1.185939326,1.192324477], allow_inf_nan=False
  )]
  redshift : Annotated[float, Field(
    ..., description="Redshift Value (required)",
    ge=-2,le=10, examples=[-0.000133202, 0.02130958], allow_inf_nan=False
  )]
  psfMag_r : Annotated[float, Field(
    ..., description="PSF Magnitude r band (required)",
    ge = -30, le = 30, examples=[16.77047, 18.43308], allow_inf_nan=False
  )]
  u : Annotated[float, Field(
    ..., description="Ultraviolet Band (required)",
    ge = -30, le = 30, examples=[16.77047, 18.43308], allow_inf_nan=False 
  )]
  g : Annotated[float, Field(
    ..., description="Green Band (required)",
    ge = -30, le = 30, examples=[16.77047, 18.43308], allow_inf_nan=False 
  )]
  r : Annotated[float, Field(
    ..., description="Red Band (required)",
    ge = -30, le = 30, examples=[16.77047, 18.43308], allow_inf_nan=False 
  )]
  i : Annotated[float, Field(
    ..., description="Near-infrared Band (required)",
    ge = -30, le = 30, examples=[16.77047, 18.43308], allow_inf_nan=False 
  )]
  z : Annotated[float, Field(
    ..., description="Infrared Band (required)",
    ge = -30, le = 30, examples=[16.77047, 18.43308], allow_inf_nan=False
  )]

  