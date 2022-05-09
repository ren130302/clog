import time
#from numba import jit
import multiprocessing as mp
import ctypes
import numpy as np
import pygame
from pygame.locals import *
import sys
from functools import lru_cache
from numba import jit
from functools import reduce

from joblib import Parallel

class clog:
    def __init__(self, pixels:int = 10, size:tuple=None, sec:float = 0.1, generation:int = -1, lg_map:list = None):
        """
        画面設定
        """
        # print("initilizing")
        pygame.init()
        pygame.display.set_caption("GAME")     
        self.pixels = mp.RawValue(ctypes.c_uint16 ,max(1, pixels)).value
        self.font = pygame.font.Font(None, 50) 
        
        if size is None :
            pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            width = pygame.display.get_surface().get_width()
            height = pygame.display.get_surface().get_height()
        else:
            width = size[0]
            height = size[1]
            
        self.width = mp.RawValue(ctypes.c_uint16 ,width).value
        self.height = mp.RawValue(ctypes.c_uint16 ,height).value
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.width_cell = mp.RawValue(ctypes.c_uint16 ,self.width//self.pixels).value
        self.height_cell = mp.RawValue(ctypes.c_uint16 ,self.height//self.pixels).value
        
        if lg_map == None:
            lg_map = np.ones((self.height_cell, self.width_cell), dtype=np.bool8)
        else :
            for y in range(self.height_cell):
                if y in range(len(lg_map),self.height_cell):
                    lg_map.append([0 for x in range(self.width_cell)])
                else:
                    for x in range(len(lg_map[y]),self.width_cell):
                        lg_map[y].append(0)
        
        self.lg_map = tuple(lg_map)
        self.next_lg_map = np.zeros((self.height_cell, self.width_cell), dtype=np.int16)
        self.coord_list = list()
        self.sec = mp.RawValue(ctypes.c_float,sec).value
        self.generation = mp.RawValue(ctypes.c_ulonglong, generation).value
        self.generation_count = mp.RawValue(ctypes.c_ulonglong, 0).value
        
        self.color_live = np.ctypeslib.as_array(mp.RawArray(ctypes.c_uint8, (66,255,66)))
        self.color_dead = np.ctypeslib.as_array(mp.RawArray(ctypes.c_uint8, (11,11,11)))
        
        self.running = mp.RawValue(ctypes.c_bool, True)
        self.fullscreen = mp.RawValue(ctypes.c_bool, True)
        self.debugmode = mp.RawValue(ctypes.c_bool, True)
        
        # print("initilized")
    
    
    
    def __toggle_running(self):
        self.running.value = not self.running.value
        # print("running=",self.running.value)
    
    def __toggle_fullscreen(self):
        self.fullscreen.value = not self.fullscreen.value
        # print("fullscreen=",self.fullscreen.value)
        if self.fullscreen.value:
            pygame.display.set_mode((self.width, self.height), FULLSCREEN)
        else:
            pygame.display.set_mode((800, 800), RESIZABLE)
    
    def __toggle_debugs(self):
        self.debugmode.value = not self.debugmode.value
        # print("debugmode=",self.debugmode.value)
        if self.debugmode.value:
            pygame.display.set_mode((self.width, self.height), FULLSCREEN)
    
    
    
    def run(self):
        """
        実行用
        """
        Parallel(n_jobs=-1)(self.mainloop())
    
    def mainloop(self):
        # print("start mainloop")
        for y in range(self.height_cell):
            for x in range(self.width_cell):
                if self.lg_map[y][x]:
                    self.coord_list.append((x, y))
        while(1):
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.__toggle_running()
                    elif event.key == K_TAB:
                        self.__toggle_debugs()
                    elif event.key == K_F11:
                        self.__toggle_fullscreen()
                    elif event.key == K_BACKSPACE:
                        pygame.display.quit()
                        pygame.quit()
                        sys.exit()
                    elif event.key == K_RETURN:
                        self.update()
            if self.generation_count >= self.generation > 0:
                self.running.value = False
            if self.debugmode.value:
                self.debug()
            if self.running.value:
                self.update()
            
            pygame.display.update()
        # print("ended mainloop")
    
    
    
    def debug(self):
        self.screen.blit(self.font.render(str(self.generation_count), True, (255,255,255)), [0, 0])
        pygame.display.update()
    
    
    
    def update(self):
        """
        マップ更新
        渡される座標が枯渇するまで繰り返し
        """
        # print("update")
        # print("generation=",self.generation_count)
        self.screen.fill(self.color_dead)
        self.next_lg_map = np.zeros((self.height_cell, self.width_cell), dtype=np.int16)
        copy_coord_list = tuple(set(self.coord_list))
        # print("coord_list=",self.coord_list)
        self.coord_list.clear()
        self.__update(copy_coord_list)
        self.lg_map = self.next_lg_map
        self.generation_count += 1
        pygame.display.update()
        if(self.sec > 0):
            time.sleep(self.sec)
    
    

    def __update(self, coord_list):
        for x, y in coord_list:
            if self.__inrange(x,y):
                self.__update_cell(x,y)
                self.__draw_cell(x,y)
    
    
    
    def __update_cell(self, x, y):
        """
        周囲８マスの状態を確認し、生きてるマスをカウント
        死んでいるセルに隣接する生きたセルがちょうど3つあれば、次の世代が誕生
        生きているセルに隣接する生きたセルが2つか3つならば、次の世代でも生存
        
        分岐最適化
        生きているセルがちょうど3つなら生存
        生きているセルに隣接する生きたセルがちょうど2つなら生存
        それ以外 -> 死滅
        
        生きているセル一つ以上で周囲の座標を追加
        """
        # print("__update_cell")
        count = 0
        cells = self.__cells(x,y)
        
        for cx,cy in cells:
            # print("__update_cell, cx=",cx, " ,cy=",cy)
            if self.__inrange(cx,cy) and self.lg_map[cy][cx] == 1:
                count += 1
        
        # print("__update_cell, count=",count)
        
        if (count > 0):
            # print("__update_cell, append cells")
            
            # self.coord_list.append((x, y))
            self.coord_list.extend(cells)
            
            # print("__update_cell, update next_lg_map")
            if (count == 3 or (self.lg_map[y][x] == 1 and count == 2)):
                self.next_lg_map[y][x] = 1
    
    
    
    def __draw_cell(self, x, y):
        """
        描画関数
        フラグが１になっているものだけを描画
        座標とマスサイズを掛け算し描画座標を算出
        """
        # print("__draw_cell")
        
        if self.lg_map[y][x]:
            # print("__draw_cell, draw cell")
            mx,my,mw,mh = self.__cell_xywh(x,y,self.pixels,self.pixels)
            pygame.draw.rect(self.screen,self.color_live,Rect(mx, my, mw, mh),0)  
    
    
    
    @lru_cache(maxsize=10000,typed=int)
    def __inrange(self, x:int, y:int):
        
        return (-1 < x < self.width_cell) and (-1 < y < self.height_cell)
    
    @staticmethod
    @lru_cache(maxsize=10000,typed=int)
    #"u8[:](u8,u8,u8,u8)",
    @jit(nopython=True, parallel=True, nogil=True)
    def __cells(x:int, y:int):
        """
        Calculate the coordinates of the eight surrounding squares from the argument
        
        Args
        ---
            x:int
                x coord(index)
            y:int
                y coord(index)
        
        Returns
        ---
            Return the eight surrounding squares
        """
        return ((x-1,y-1),(x,y-1),(x+1,y-1),(x-1,y),(x+1,y),(x-1,y+1),(x,y+1),(x+1,y+1))
    
    @staticmethod
    @lru_cache(maxsize=10000,typed=int)
    #"u8[:](u8,u8,u8,u8)",
    @jit(nopython=True, parallel=True, nogil=True)
    def __cell_xywh(x:int, y:int, w:int, h:int):
        """
        Calculate coordinates from arguments.
        
        Args
        ---
            x:int
                x coord(index)
            y:int
                y coord(index)
            width:int 
                Specify the width in pixels
            height:int 
                Specify the height in pixels
        
        Returns
        ---
            Tuple of coordinates and size to be displayed
        """
        return x*w, y*h, w, h
    
    
    
    def __compress(self):
        cl_map = []
        
        for y in range(self.height_cell):
            cl_map.append([])
            cell = self.lg_map[y][0]
            idx = 0
            for x in range(self.width_cell):
                try:
                    cl_map[y][idx][1] += 1
                    if cell != self.lg_map[y][x]:
                        idx += 1
                        cell = self.lg_map[y][x]
                except IndexError:
                    cl_map[y].append([cell,1])


class GET():
    def __init__(self, dead, live, map):
        str_line = map.split()
        
        lg_map = []
        
        for y in range(len(str_line)):
            lg_map.append([])
            for x in range(len(str_line[y])):
                if str_line[y][x] == dead:
                    lg_map[y].append(0)
                elif str_line[y][x] == live:
                    lg_map[y].append(1)
        
        self.lg_map = lg_map
    def ret(self):
        return self.lg_map
        

lg_map = GET(dead="□", live="■",map= 
"""
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■■□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■■□□□□■■□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□■■■□□■□■■□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□■□■□□■□■□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□□■□■□■□■□■■□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□□■□■□□□■■□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■■■□□□□□■□■□□□□■□□□■□■■■□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□■■□■□■■■□■■□□□□□□□□□■■□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□□□■■□□□□□■□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□■■□■□□■□□■□■■□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□■□■□■□■□■□□□□□■■■■□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□■■□■□□■□□■□□■■□■□■■□□□■□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□□□■■□□□■□■□■□□□■■□□□□□■□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□■■□■□■■□□■□□■□□■□■■□□■□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■■■□□□□□■□■□■□■□■□■□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■□■□□■□□■□■■□□■□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□□□■■□□□□□■□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■□□□□□□□□□■■□■■■□■□■■□□□■□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■■□■□□□■□□□□■□■□□□□□■■■■□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■□□□■□■□□□□■□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■□■□■□■□■□□□□■□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□■□□■□■□□□■□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■□■□□■■■□□■□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■□□□□■■■□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■■■□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□■□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□""").ret()


# lg_map = GET(dead="□", live="■",map= 
# """
# □□□□□□
# □□■■□□
# □□■■□□""").ret()


#insta = clog(pixels=2,lg_map=lg_map, width=100,height=100, sec=0, generation =-1)
insta = clog(pixels=10,lg_map=lg_map, sec=0, generation =-1)


insta.run()
