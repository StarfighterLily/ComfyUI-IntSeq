from PIL import Image, ImageDraw
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import math
import random

def remap( val, min_val, max_val, min_map, max_map ):
    if max_val == min_val:
        return min_map
    return ( val - min_val ) / ( max_val - min_val ) * ( max_map - min_map ) + min_map
    
def conv_pil_tensor( img ):
    return ( torch.from_numpy( np.array( img ).astype( np.float32 ) / 255.0 ).unsqueeze( 0 ), )

def clamp( val, min_val, max_val ):
    return max( min_val, min( val, max_val ) )

class IntSeqImage:
    @classmethod
    def INPUT_TYPES( cls ):
        return {
            "required": {
                "width": ( "INT", { "default": 512, "min": 128, "max": 8192, "step": 8, "tooltip": "Width of the image" } ),
                "height": ( "INT", { "default": 512, "min": 128, "max": 8192, "step": 8, "tooltip": "Height of the image" } ),
                "sequence": ( "STRING", { "multiline": True, "default": "", "tooltip": "Enter a list of numbers separated by commas" } ),
                "method": ( [ "RGB", "angle and length", "run and turn", "meander", "cellular_automaton" ], { "default": "RGB" } ),
                "rule": ( "INT", { "default": 30, "min": 0, "max": 255, "step": 1, "tooltip": "[Cellular Automaton] The rule to apply (0-255)" } ),
                "color_offset": ( "FLOAT", { "default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "0 - 1" } ),
                "value_min": ( "INT", { "default": -1, "min": -1, "max": 255, "step": 1, "tooltip": " -1 - 255" } ),
                "value_max": ( "INT", { "default": -1, "min": -1, "max": 255, "step": 1, "tooltip": " -1 - 255" } ),
                "red_min": ( "INT", { "default": -1, "min": -1, "max": 255, "step": 1, "tooltip": " -1 - 255" } ),
                "red_max": ( "INT", { "default": -1, "min": -1, "max": 255, "step": 1, "tooltip": " -1 - 255" } ),
                "green_min": ( "INT", { "default": -1, "min": -1, "max": 255, "step": 1, "tooltip": " -1 - 255" } ),
                "green_max": ( "INT", { "default": -1, "min": -1, "max": 255, "step": 1, "tooltip": " -1 - 255" } ),
                "blue_min": ( "INT", { "default": -1, "min": -1, "max": 255, "step": 1, "tooltip": " -1 - 255" } ),
                "blue_max": ( "INT", { "default": -1, "min": -1, "max": 255, "step": 1, "tooltip": " -1 - 255" } ),
                "angle_scale": ( "FLOAT", { "default": 1.0, "min": -360.0, "max": 360.0, "step": 0.1, "tooltip": "[Angle/Length] Multiplier for turn angle" } ),
                "length_scale": ( "FLOAT", { "default": 10.0, "min": -200.0, "max": 200.0, "step": 0.1, "tooltip": "[Angle/Length] Multiplier for line length [Meander] Pixel distance multiplier" } ),
                "line_width": ( "INT", { "default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "[Angle/Length, Run/Turn] Width of the drawn line" } ),
                "start_x": ( "INT", { "default": 0, "min": 0, "max": 1024, "step": 1, "tooltip": "Starting X" } ),
                "start_y": ( "INT", { "default": 0, "min": 0, "max": 1024, "step": 1, "tooltip": "Starting Y" } ),
                "boundary_behavior": ( [ "clamp", "wrap", "bounce", "none" ], { "default": "clamp", "tooltip": "[Angle/Length, Run/Turn, Meander] How to handle the drawing point at image boundaries" } )
            }
        }

    RETURN_TYPES = ( "IMAGE", )
    FUNCTION = "generate_image"
    CATEGORY = "IntSeq/image"

    def generate_image( self, width, height, sequence, method, rule, color_offset, value_min, value_max, red_min, red_max, green_min, green_max, blue_min, blue_max, angle_scale, length_scale, line_width, start_x, start_y, boundary_behavior ):
        img_mode = "RGB" if method in [ "RGB", "angle and length", "run and turn", "meander", "cellular_automaton" ] else "L"
        
        lv = 0 if value_min == -1 else value_min
        mv = 255 if value_max == -1 else value_max
        lr = lv if red_min == -1 else red_min
        mr = mv if red_max == -1 else red_max
        lg = lv if green_min == -1 else green_min
        mg = mv if green_max == -1 else green_max
        lb = lv if blue_min == -1 else blue_min
        mb = mv if blue_max == -1 else blue_max

        try:
            values = [ float( x.strip() ) for x in sequence.split( ',' ) if x.strip() ]
        except ValueError:
            raise ValueError( "Warning: [IntSeqNoise] Could not parse all values. Please ensure it's a comma-separated list of numbers." )

        if not values:
            return ( torch.zeros( ( 1, height, width, 3 ), dtype=torch.float32 ), )
        
        outimage = Image.new( img_mode, ( width, height ), (0,0,0) )
        draw = ImageDraw.Draw( outimage )

        if method == "RGB":
            value_index = 0
            for y in range( height ):
                for x in range( width ):
                    nv = values[ value_index % len( values ) ]
                    nv = clamp( nv, lv, mv )
                    nr = int( remap( nv, lv, mv, lr, mr ) )
                    ng = int( remap( ( nv + color_offset * ( mv - lv ) ) % ( mv - lv + 1 ) + lv, lv, mv, lg, mg ) )
                    nb = int( remap( ( nv + 2 * color_offset * ( mv - lv ) ) % ( mv - lv + 1 ) + lv, lv, mv, lb, mb ) )
                    outimage.putpixel( ( x, y ), ( nr, ng, nb ) )
                    value_index += 1

        elif method == "cellular_automaton":
            rule_bits = format( rule, '08b' )
            
            current_row = [ 0 ] * width
            for i in range( width ):
                val = values[ i % len( values ) ]
                current_row[ i ] = int( remap( val, min( values ), max( values ), 0, 1.99 ) ) % 2

            color0 = ( lr, lg, lb )
            color1 = ( mr, mg, mb )

            for y in range( height ):
                next_row = [ 0 ] * width
                for x in range( width ):
                    state = current_row[ x ]
                    color = color1 if state == 1 else color0
                    outimage.putpixel( ( x, y ), color )
                    
                    if y < height - 1:
                        left = current_row[ ( x - 1 + width ) % width ]
                        center = current_row[ x ]
                        right = current_row[ ( x + 1 ) % width ]
                        
                        pattern_index = 7 - ( left * 4 + center * 2 + right )
                        new_state = int( rule_bits[ pattern_index ] )
                        next_row[ x ] = new_state
                
                current_row = next_row

        elif method == "meander":
            current_x = start_x
            current_y = start_y

            for i in values:
                color_val = clamp( i, lv, mv )
                nr = int( remap( color_val, lv, mv, lr, mr ) )
                ng = int( remap( ( color_val + color_offset * ( mv - lv ) ) % ( mv - lv + 1 ) + lv, lv, mv, lg, mg ) )
                nb = int( remap( ( color_val + 2 * color_offset * ( mv - lv ) ) % ( mv - lv + 1 ) + lv, lv, mv, lb, mb ) )
                pixel_color = ( nr, ng, nb )

                direction = int( remap( i, lv, mv, 0, 7 ) )
                if direction == 0:    # Up
                    current_y -= 1 * length_scale
                elif direction == 1:  # Up-Right
                    current_y -= 1 * length_scale
                    current_x += 1 * length_scale
                elif direction == 2:  # Right
                    current_x += 1 * length_scale
                elif direction == 3:  # Down-Right
                    current_y += 1 * length_scale
                    current_x += 1 * length_scale
                elif direction == 4:  # Down
                    current_y += 1 * length_scale
                elif direction == 5:  # Down-Left
                    current_y += 1 * length_scale
                    current_x -= 1 * length_scale
                elif direction == 6:  # Left
                    current_x -= 1 * length_scale
                elif direction == 7:  # Up-Left
                    current_y -= 1 * length_scale
                    current_x -= 1 * length_scale

                draw_x, draw_y = current_x, current_y
                if boundary_behavior == "clamp":
                    draw_x = clamp( current_x, 0, width - 1 )
                    draw_y = clamp( current_y, 0, height - 1 )
                elif boundary_behavior == "wrap":
                    draw_x = current_x % width
                    draw_y = current_y % height
                elif boundary_behavior == "bounce":
                    if not 0 <= current_x < width:
                        current_x = clamp( current_x, 0, width - 1 )
                    if not 0 <= current_y < height:
                        current_y = clamp( current_y, 0, height - 1 )
                        draw_x, draw_y = current_x, current_y
        
                if 0 <= draw_x < width and 0 <= draw_y < height:
                    outimage.putpixel( ( int( draw_x ), int( draw_y ) ), pixel_color )
        
        elif method == "angle and length":
            current_x = start_x
            current_y = start_y
            current_angle = 0.0

            for i in range( 0, len( values ) - 1, 2 ):
                segment_length = values[ i ] * length_scale
                turn_angle = values[ i + 1 ] * angle_scale
                current_angle += turn_angle
                nv = clamp( values[ i + 1 ], lv, mv )
                nr = int( remap( nv, lv, mv, lr, mr ) )
                ng = int( remap( ( nv + color_offset * ( mv - lv ) ) % ( mv - lv + 1 ) + lv, lv, mv, lg, mg ) )
                nb = int (remap( ( nv + 2 * color_offset * ( mv - lv ) ) % ( mv - lv + 1 ) + lv, lv, mv, lb, mb ) )
                line_color = ( nr, ng, nb )
                rad_angle = math.radians( current_angle )
                end_x = start_x + segment_length * math.cos( rad_angle )
                end_y = start_y + segment_length * math.sin( rad_angle )
                draw.line( [ ( start_x, start_y ), ( end_x, end_y ) ], fill=line_color, width=line_width )
                
                if boundary_behavior == "clamp":
                    start_x = clamp( end_x, 0, width - 1 )
                    start_y = clamp( end_y, 0, height - 1 )
                elif boundary_behavior == "wrap":
                    start_x = end_x % width
                    start_y = end_y % height
                elif boundary_behavior == "bounce":
                    bounced = False
                    if end_x < 0 or end_x >= width:
                        current_angle = 180 - current_angle
                        bounced = True
                    if end_y < 0 or end_y >= height:
                        current_angle = 360 - current_angle
                        bounced = True
                    if bounced:
                        start_x = clamp( end_x, 0, width - 1 )
                        start_y = clamp( end_y, 0, height - 1 )
                    else:
                        start_x, start_y = end_x, end_y
                else:
                    start_x, start_y = end_x, end_y

        elif method == "run and turn":
            current_x = start_x
            current_y = start_y

            for i in range( 0, len( values ) - 1, 2 ):
                segment_length = values[ i ] * length_scale
                turn_angle = values[ int( i + 1 ) ] * angle_scale
                turn_angle = remap( turn_angle, lv, mv, 0, 360 )
                nv = clamp( values[ i ], lv, mv )
                nr = int( remap( nv, lv, mv, lr, mr ) )
                ng = int( remap( ( nv + color_offset * ( mv - lv ) ) % ( mv - lv + 1 ) + lv, lv, mv, lg, mg ) )
                nb = int (remap( ( nv + 2 * color_offset * ( mv - lv ) ) % ( mv - lv + 1 ) + lv, lv, mv, lb, mb ) )
                line_color = ( nr, ng, nb )
                rad_angle = math.radians( turn_angle )
                end_x = start_x + segment_length * math.cos( rad_angle )
                end_y = start_y + segment_length * math.sin( rad_angle )
                draw.line( [ ( start_x, start_y ), ( end_x, end_y ) ], fill=line_color, width=line_width )
                
                if boundary_behavior == "clamp":
                    start_x = clamp( end_x, 0, width - 1 )
                    start_y = clamp( end_y, 0, height - 1 )
                elif boundary_behavior == "wrap":
                    start_x = end_x % width
                    start_y = end_y % height
                elif boundary_behavior == "bounce":
                    bounced = False
                    if end_x < 0 or end_x >= width:
                        turn_angle = 180 - turn_angle
                        bounced = True
                    if end_y < 0 or end_y >= height:
                        turn_angle = 360 - turn_angle
                        bounced = True
                    if bounced:
                        start_x = clamp( end_x, 0, width - 1 )
                        start_y = clamp( end_y, 0, height - 1 )
                    else:
                        start_x, start_y = end_x, end_y
                else:
                    start_x, start_y = end_x, end_y

        return conv_pil_tensor( outimage )

class IntSeqSigmas:
    @classmethod
    def INPUT_TYPES( cls ):
        return {
            "required": {
                "sequence": ( "STRING", { "multiline": True, "default": "", "tooltip": "Enter a list of numbers separated by commas" } ),
                "count": ( "INT", { "default": 0, "min": 0, "max": 4096, "step": 1, "tooltip": "How many numbers to use from the start of the sequence (0 for all)" } ),
                "new_minimum": ( "FLOAT", { "default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01, "tooltip": "The new lowest value in the output range" } ),
                "new_maximum": ( "FLOAT", { "default": 20.0, "min": -10000.0, "max": 10000.0, "step": 0.01, "tooltip": "The new highest value in the output range" } ),
                "reverse": ( "BOOLEAN", { "default": True } ),
                "sort_order": ( [ "no sort", "highest to lowest", "lowest to highest" ], { "default": "no sort" } ),
            },
        }

    RETURN_TYPES = ( "SIGMAS", )
    RETURN_NAMES = ( "SIGMAS", )
    FUNCTION = "map_sequence"
    CATEGORY = "IntSeq/sigmas"

    def map_sequence( self, sequence, count, new_minimum, new_maximum, reverse, sort_order ):
        try:
            value_list = [ float( x.strip() ) for x in sequence.split( ',' ) if x.strip() ]
        except ValueError:
            print( "Warning: [IntSeqSigmas] Could not parse all values. Please ensure it's a comma-separated list of numbers." )
            return ( torch.empty( 0 ), )

        if not value_list:
            print( "Warning: [IntSeqSigmas] Input value list is empty." )
            return ( torch.empty( 0 ), )

        val_subset = value_list[ :count ] if 0 < count <= len( value_list ) else value_list

        if sort_order == "highest to lowest":
            val_subset.sort( reverse=True )
        elif sort_order == "lowest to highest":
            val_subset.sort( reverse=False )

        if reverse:
            val_subset.reverse( )

        if not val_subset:
            return ( torch.empty( 0 ), )

        original_minimum = min( val_subset )
        original_maximum = max( val_subset )

        mapped_values = [ remap( x, original_minimum, original_maximum, new_minimum, new_maximum ) for x in val_subset ]
        sigmas_tensor = torch.tensor( mapped_values, dtype=torch.float32, device='cpu' )

        return ( sigmas_tensor, )

class IntSeqPlotter:
    @classmethod
    def INPUT_TYPES( cls ):
        return {
            "required": {
                "sequence": ( "STRING", { "multiline": True, "default": "", "tooltip": "Enter a list of numbers separated by commas" } ),
                "title": ( "STRING", { "default": "Sequence Plot " } ),
                "xlabel": ( "STRING", { "default": "Index" } ),
                "ylabel": ( "STRING", { "default": "Value" } ),
            }
        }

    RETURN_TYPES = ( "IMAGE", )
    FUNCTION = "plot_sequence"
    CATEGORY = "IntSeq/plot"

    def plot_sequence( self, sequence, title, xlabel, ylabel ):
        try:
            values = [ float( x.strip() ) for x in sequence.split( ',' ) if x.strip() ]
        except ValueError:
            raise ValueError( "Warning: [IntSeqPlotter] Could not parse all values. Please ensure it's a comma-separated list of numbers." )

        if not values:
            return ( torch.zeros( ( 1, 100, 100, 3 ), dtype=torch.float32 ), )

        fig, ax = plt.subplots( )
        ax.plot( values )
        ax.set_title( title )
        ax.set_xlabel( xlabel )
        ax.set_ylabel( ylabel )
        ax.grid( True )
        
        buf = io.BytesIO( )
        plt.savefig( buf, format='png' )
        plt.close( fig )
        buf.seek( 0 )
        
        image = Image.open( buf )
        image = image.convert( "RGB" )
        image_np = np.array( image ).astype( np.float32 ) / 255.0
        image_tensor = torch.from_numpy( image_np ).unsqueeze( 0 )
        
        return ( image_tensor, )

class IntSeqWave:
    @classmethod
    def INPUT_TYPES( cls ):
        return {
            "required": {
                "type": ( [ "sine", "cosine", "tangent", "cotangent", "sawtooth", "triangle", "sigmoid", "square" ], { "default": "sine" } ),
                "length": ( "INT", { "default": 100, "min": 1, "max": 10000, "step": 1, "tooltip": "Number of values in the wave LUT" } ),
                "amplitude": ( "FLOAT", { "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01, "tooltip": "Amplitude of the wave" } ),
                "frequency": ( "FLOAT", { "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip": "Frequency of the wave (number of cycles)" } ),
                "phase_offset": ( "FLOAT", { "default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1, "tooltip": "Phase offset in degrees" } ),
                "vertical_offset": ( "FLOAT", { "default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01, "tooltip": "Vertical offset of the wave" } ),
                "slope": ( "FLOAT", { "default": 0.0, "min": -29.0, "max": 29.0, "step": 0.01, "tooltip": "[Sigmoid]Slope of the wave" } ),
                "duty_cycle": ( "FLOAT", { "default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01, "tooltip": "[Triangle Sawtooth Square]Duty cycle of the wave" } ),
            }
        }

    RETURN_TYPES = ( "STRING", )
    RETURN_NAMES = ( "SEQUENCE", )
    FUNCTION = "generate_wave_sequence"
    CATEGORY = "IntSeq/generator"

    def generate_wave_sequence( self, type, length, amplitude, frequency, phase_offset, vertical_offset, slope, duty_cycle ):
        values = []
        phase_rad = math.radians( phase_offset )

        for i in range( length ):
            t = ( i / length ) * ( 2 * math.pi * frequency ) + phase_rad
            
            if type == "sine":
                value = amplitude * math.sin( t ) + vertical_offset
            elif type == "cosine":
                value = amplitude * math.cos( t ) + vertical_offset
            elif type == "tangent":
                try:
                    tan_val = math.tan( t )
                    value = amplitude * math.atan( tan_val ) * ( 2 / math.pi ) + vertical_offset
                except ValueError:
                    value = float( 'nan' )
            elif type == "cotangent":
                try:
                    cot_val = 1 / math.tan( t )
                    value = amplitude * math.atan( cot_val ) * ( 2 / math.pi ) + vertical_offset
                except ( ValueError, ZeroDivisionError ):
                    value = float( 'nan' )
            elif type == "sawtooth":
                normalized_t = ( t / ( 2 * math.pi ) ) % 1
                if normalized_t < duty_cycle:
                    value = amplitude * ( normalized_t / duty_cycle ) - amplitude / 2
                else:
                    value = amplitude * ( ( normalized_t - duty_cycle ) / ( 1 - duty_cycle ) ) - amplitude / 2
                value += vertical_offset
            elif type == "triangle":
                normalized_t = ( t / ( 2 * math.pi ) ) % 1
                if normalized_t < duty_cycle:
                    value = amplitude * ( 2 * normalized_t / duty_cycle - 1 )
                else:
                    value = amplitude * ( 1 - 2 * ( normalized_t - duty_cycle ) / ( 1 - duty_cycle ) )
                value += vertical_offset
            elif type == "sigmoid":
                sigmoid_input = remap( t, 0, 2 * math.pi * frequency, -6 * frequency, 6 * frequency ) * slope
                value = amplitude * ( 1 / ( 1 + math.exp( -sigmoid_input ) ) ) + vertical_offset - amplitude / 2
            elif type == "square":
                normalized_t = ( t / ( 2 * math.pi ) ) % 1
                if normalized_t < duty_cycle:
                    value = amplitude + vertical_offset
                else:
                    value = -amplitude + vertical_offset
            else:
                value = 0
            
            values.append( value )

        if any( math.isnan( v ) for v in values ):
            print( f"Warning: [IntSeqWave] Generated NaN values for wave type '{type}'. These will be converted to 0." )
            values = [ v if not math.isnan( v ) else 0 for v in values ]


        return ( ",".join( map( str, values ) ), )

# --- Node Mappings ---

NODE_CLASS_MAPPINGS = {
    "IntSeqImage": IntSeqImage,
    "IntSeqSigmas": IntSeqSigmas,
    "IntSeqPlotter": IntSeqPlotter,
    "IntSeqWave": IntSeqWave
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntSeqImage": "Integer Sequence Image",
    "IntSeqSigmas": "Integer Sequence Sigmas",
    "IntSeqPlotter": "Integer Sequence Plotter",
    "IntSeqWave": "Integer Sequence Wave"
}