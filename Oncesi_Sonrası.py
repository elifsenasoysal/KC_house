import leafmap.foliumap as lm 
img1=r'/Users/elifsenasoysal/Desktop/btk/KC_house/061b68c6-f1c5-4c03-8acd-11a351960fec.JPG'
img2=r'/Users/elifsenasoysal/Desktop/btk/KC_house/b55ba2c8-baee-4038-bad6-167905bd6aeb.JPG'
lm.image_comparison(img1,img2,
                     label1='Once',
                     label2='Sonra',
                     starting_position=50,
                     out_html='orman_once_sonra.html')