import sys, os, argparse, cv2, glob, math, random, json

FileDirPath = os.path.dirname(os.path.realpath(__file__))
from tk3dv import pyEasel
from PyQt5.QtWidgets import QApplication
import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent

from EaselModule import EaselModule
from Easel import Easel
import numpy as np
import OpenGL.GL as gl
from tk3dv.nocstools import datastructures as ds
from tk3dv.nocstools import obj_loader

from palettable.tableau import Tableau_20, BlueRed_12, ColorBlind_10, GreenOrange_12
from palettable.cartocolors.diverging import Earth_2
import calibration
from tk3dv.common import drawing, utilities
from tk3dv.extern import quaternions

from sklearn.neighbors import NearestNeighbors

class StrobeNetModule(EaselModule):
    def __init__(self):
        super().__init__()

    def init(self, InputArgs=None):
        self.Parser = argparse.ArgumentParser(description='StrobeNetModule to results for the StrobeNet project.',
                                              fromfile_prefix_chars='@')
        ArgGroup = self.Parser.add_argument_group()
        ArgGroup.add_argument('--nocs-maps', nargs='+', help='Specify input NOCS maps. * globbing is supported.',
                              required=True)
        ArgGroup.add_argument('--colors', nargs='+',
                              help='Specify RGB images corresponding to the input NOCS maps. * globbing is supported.',
                              required=False)
        ArgGroup.add_argument('--models', nargs='+',
                              help='Specify OBJ models to load additionally. * globbing is supported.',
                              required=False)
        ArgGroup.add_argument('--seg-maps', nargs='+',
                              help='Specify segmentation maps to load. * globbing is supported.',
                              required=False)
        ArgGroup.add_argument('--jt-pos', nargs='+',
                              help='Specify joint positions to load. * globbing is supported.',
                              required=False)
        ArgGroup.add_argument('--jt-ang', nargs='+',
                              help='Specify joint angles to load. * globbing is supported.',
                              required=False)

        self.Args, _ = self.Parser.parse_known_args(InputArgs)
        if len(sys.argv) <= 1:
            self.Parser.print_help()
            exit()

        self.NOCSMaps = []
        self.NOCS = []
        self.AnimatableNOCS = [] # For point cloud animations
        self.ReferenceNOCS = []  # For reference
        self.SegmentIndices = []
        self.UniqueSegmentColors = []
        self.ValidAnimatableSegments = []
        self.OBJModels = []
        self.AnimatableOBJModels = []
        self.M2NIndices = []
        self.JtPos = []
        self.JtAng = []
        self.JtNum = 0
        self.SegMaps = []
        self.PointSize = 5
        self.AxisWidth = 5
        self.AxisHalfLength = 0.1
        self.JtRadius = 0.01
        self.AngleIncrem = 0.0001
        self.CurrentAngle = 0

        sys.stdout.flush()
        self.nNM = 0
        self.SSCtr = 0
        self.takeSS = False
        self.showNOCS = True
        self.showBB = False
        self.showPoints = True
        self.showWireFrame = False
        self.showOBJModels = True
        self.showJointPos = True
        self.showJointAng = True
        self.showAnimation = True
        self.loadData()

    def drawNOCS(self, lineWidth=2.0, ScaleX=1, ScaleY=1, ScaleZ=1, OffsetX=0, OffsetY=0, OffsetZ=0):
        gl.glPushMatrix()

        gl.glScale(ScaleX, ScaleY, ScaleZ)
        gl.glTranslate(OffsetX, OffsetY, OffsetZ)  # Center on cube center
        drawing.drawUnitWireCube(lineWidth, True)

        gl.glPopMatrix()

    @staticmethod
    def getFileNames(InputList):
        if InputList is None:
            return []
        FileNames = []
        for File in InputList:
            if '*' in File:
                GlobFiles = glob.glob(File, recursive=False)
                GlobFiles.sort()
                FileNames.extend(GlobFiles)
            else:
                FileNames.append(File)

        return FileNames

    def resizeAndPad(self, Image):
        # SquareUpSize = min(self.ImageSize[0], self.ImageSize[1])
        # TODO
        print('[ INFO ]: Original input size ', Image.shape)
        Image = cv2.resize(Image, self.ImageSize, interpolation=cv2.INTER_NEAREST)
        print('[ INFO ]: Input resized to ', Image.shape)
        sys.stdout.flush()

        return Image

    def loadData(self):
        Palette = ColorBlind_10
        NMFiles = self.getFileNames(self.Args.nocs_maps)
        ColorFiles = [None] * len(NMFiles)
        if self.Args.colors is not None:
            ColorFiles = self.getFileNames(self.Args.colors)

        for (NMF, CF) in zip(NMFiles, ColorFiles):
            NOCSMap = cv2.imread(NMF, -1)
            NOCSMap = NOCSMap[:, :, :3]  # Ignore alpha if present
            NOCSMap = cv2.cvtColor(NOCSMap, cv2.COLOR_BGR2RGB)  # IMPORTANT: OpenCV loads as BGR, so convert to RGB
            self.ImageSize = (NOCSMap.shape[1], NOCSMap.shape[0])
            CFIm = None
            if CF is not None:
                CFIm = cv2.imread(CF)
                CFIm = cv2.cvtColor(CFIm, cv2.COLOR_BGR2RGB)  # IMPORTANT: OpenCV loads as BGR, so convert to RGB
                if CFIm.shape != NOCSMap.shape:  # Re-size only if not the same size as NOCSMap
                    CFIm = cv2.resize(CFIm, (NOCSMap.shape[1], NOCSMap.shape[0]),
                                      interpolation=cv2.INTER_CUBIC)  # Ok to use cubic interpolation for RGB
            NOCS = ds.NOCSMap(NOCSMap, RGB=CFIm)
            AnimatableNOCS = ds.NOCSMap(NOCSMap, RGB=CFIm)
            self.NOCSMaps.append(NOCSMap)
            self.NOCS.append(NOCS)
            self.AnimatableNOCS.append(AnimatableNOCS)

        self.nNM = len(NMFiles)
        self.activeNMIdx = self.nNM  # len(NMFiles) will show all

        if self.Args.jt_pos is not None and self.Args.jt_ang is not None and self.Args.seg_maps is not None:
            PosFiles = self.getFileNames(self.Args.jt_pos)
            AngFiles = self.getFileNames(self.Args.jt_ang)
            SegFiles = self.getFileNames(self.Args.seg_maps)

            for idx, (PosF, AngF, SegF) in enumerate(zip(PosFiles, AngFiles, SegFiles)):
                JointPos = np.loadtxt(PosF)
                JointAng  = np.loadtxt(AngF)
                if len(JointPos.shape) == 1:
                    JointPos = np.expand_dims(JointPos, axis=0)
                    JointAng = np.expand_dims(JointAng, axis=0)                
                self.JtNum = JointAng.shape[0]
                SegMap = cv2.imread(SegF, -1)
                self.JtPos.append(JointPos)
                self.JtAng.append(JointAng)
                self.SegMaps.append(SegMap)
                ReferenceNOCS = ds.NOCSMap(self.NOCSMaps[idx], RGB=SegMap)
                self.ReferenceNOCS.append(ReferenceNOCS)
                UniqueSegmentColors = np.unique(SegMap.reshape(-1, SegMap.shape[-1]), axis=0)[1:]  # Exclude the first one which is black
                
                # HACK, drop background (blue)                
                bg_idx = np.where((UniqueSegmentColors == np.array([128, 0, 0])).all(axis=1))
                # print(UniqueSegmentColors, UniqueSegmentColors.shape)
                UniqueSegmentColors = np.delete(UniqueSegmentColors, bg_idx, axis=0)
                # print(UniqueSegmentColors)
                self.UniqueSegmentColors.append(UniqueSegmentColors)
                # self.ValidAnimatableSegments.append(UniqueSegmentColors[[2, 0], :])  # HACK, TODO, for glasses only
                self.ValidAnimatableSegments.append(UniqueSegmentColors[[self.JtNum, 0], :])  # HACK, TODO, for glasses only
                # print('ValidAnimatableSegments:', ValidAnimatableSegments) # HACK TODO, for glasses only

        # Load OBJ models
        ModelFiles = self.getFileNames(self.Args.models)
        for idx, MF in enumerate(ModelFiles):
            LoadedOBJ = obj_loader.Loader(MF, isNormalize=False) # Should be False
            AnimatableLoadedOBJ = obj_loader.Loader(MF, isNormalize=False)  # Should be False
            # Normalize model vertices to at NOCS center
            VerticesNP = np.asarray(LoadedOBJ.vertices)
            LoadedOBJ.vertices = VerticesNP + 0.5
            LoadedOBJ.Colors = np.asarray(LoadedOBJ.vertices)
            print('[ INFO ]: Custom normalization')
            if len(self.ReferenceNOCS) != 0:
                # Find NN
                # print('self.ReferenceNOCS.Points', self.ReferenceNOCS[idx].Points.shape)
                # print('LoadedOBJ.vertices', LoadedOBJ.vertices.shape)
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(self.ReferenceNOCS[idx].Points)
                _, M2NIndices = nbrs.kneighbors(LoadedOBJ.vertices)
                self.M2NIndices.append(M2NIndices)
                LoadedOBJ.Colors = self.NOCS[idx].Colors[M2NIndices]
                # print(distances.shape)
            AnimatableLoadedOBJ.vertices = LoadedOBJ.vertices
            AnimatableLoadedOBJ.Colors = LoadedOBJ.Colors
            LoadedOBJ.update()
            AnimatableLoadedOBJ.update()
            self.OBJModels.append(LoadedOBJ)
            self.AnimatableOBJModels.append(AnimatableLoadedOBJ)

    def step(self):
        pass

    def createAxisAngleRotationMatrix(self, axis, angle):
        U = axis / np.linalg.norm(axis)
        t = angle # Radians

        ux = U[0]
        uy = U[1]
        uz = U[2]
        ux2 = U[0]*U[0]
        uy2 = U[1]*U[1]
        uz2 = U[2]*U[2]
        cost = math.cos(t)
        sint = math.sin(t)

        R = np.identity(3)
        R[0, 0] = cost + (ux2*(1-cost))
        R[0, 1] = (ux*uy*(1-cost)) - (uz*sint)
        R[0, 2] = (ux*uz*(1-cost)) + (uy*sint)

        R[1, 0] = (uy*ux*(1-cost)) + (uz*sint)
        R[1, 1] = cost + (uy2*(1-cost))
        R[1, 2] = (uy*uz*(1-cost)) - (ux*sint)

        R[2, 0] = (uz*ux*(1-cost)) - (uy*sint)
        R[2, 1] = (uz*uy*(1-cost)) + (ux*sint)
        R[2, 2] = cost + (uz2*(1-cost))

        return R

    def rotateNOCS(self, AnimatableNOCS, ReferenceNOCS, SegmentColor, JointPosition, Axis, Angle):
        # print(np.max(ReferenceNOCS.Colors))
        SegmentIndices = np.where(np.all(ReferenceNOCS.Colors == SegmentColor/255, axis=-1))
        # print(len(AnimatableNOCS))
        # print(SegmentIndices)
        AnimatableNOCS.Points[SegmentIndices] = ((AnimatableNOCS.Points[SegmentIndices]-JointPosition) @ self.createAxisAngleRotationMatrix(Axis, Angle)) + JointPosition
        # AnimatableNOCS.Points[SegmentIndices] = (self.createAxisAngleRotationMatrix(Axis, Angle)@(AnimatableNOCS.Points[SegmentIndices].T)).T
        AnimatableNOCS.update()

    def rotateMesh(self, AnimatableMesh, ReferenceNOCS, M2NIndices, SegmentColor, JointPosition, Axis, Angle):
        NOCS2PartIdx = np.where(np.all(ReferenceNOCS.Colors == SegmentColor/255, axis=-1))
        Mesh2PartIdx = np.where(np.in1d(M2NIndices, NOCS2PartIdx))
        # print(NOCS2PartIdx[0].shape)
        # print(Mesh2PartIdx[0].shape)
        # print(Mesh2PartIdx)
        # exit()

        AnimatableMesh.vertices[Mesh2PartIdx] = ((AnimatableMesh.vertices[Mesh2PartIdx]-JointPosition) @ self.createAxisAngleRotationMatrix(Axis, Angle)) + JointPosition
        AnimatableMesh.update()


    def animate(self):        
        for idx, (JP, JA, SegMap, ValidAnimatableSegments) in enumerate(zip(self.JtPos, self.JtAng, self.SegMaps, self.ValidAnimatableSegments)):
            self.CurrentAngle += self.AngleIncrem
            self.CurrentAngle = self.CurrentAngle%math.pi
            # print('self.CurrentAngle', self.CurrentAngle)
            print(JP.shape)
            for r in range(JP.shape[0]):                
                JointPosition = JP[r, :]
                DefaultAngle = np.linalg.norm(JA[r, :])
                Axis = JA[r, :] / DefaultAngle
                SegmentColor = ValidAnimatableSegments[r]                                
                self.rotateNOCS(self.AnimatableNOCS[idx], self.ReferenceNOCS[idx], SegmentColor, JointPosition, Axis, Angle=self.CurrentAngle)
                self.rotateMesh(self.AnimatableOBJModels[idx], self.ReferenceNOCS[idx], self.M2NIndices[idx], SegmentColor, JointPosition, Axis, Angle=self.CurrentAngle)

    def draw(self):
        self.animate()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        ScaleFact = 500
        gl.glTranslate(-ScaleFact / 2, -ScaleFact / 2, -ScaleFact / 2)
        gl.glScale(ScaleFact, ScaleFact, ScaleFact)

        for Idx, (NOCS, AnimatableNOCS) in enumerate(zip(self.NOCS, self.AnimatableNOCS)):
            if self.activeNMIdx != self.nNM:
                if Idx != self.activeNMIdx:
                    continue

            DispNOCS = NOCS
            if self.showAnimation == True:
                DispNOCS = AnimatableNOCS

            if self.showPoints:
                DispNOCS.draw(self.PointSize)
            else:
                DispNOCS.drawConn(isWireFrame=self.showWireFrame)
            if self.showBB:
                DispNOCS.drawBB()

        if self.showNOCS:
            self.drawNOCS(lineWidth=5.0)

        if self.showOBJModels:
            Models = self.OBJModels
            if self.showAnimation and self.AnimatableOBJModels is not None:
                Models = self.AnimatableOBJModels
            if Models is not None:
                for OM in Models:
                    OM.draw(isWireFrame=self.showWireFrame)

        if self.showJointPos:
            for (JP, JA) in zip(self.JtPos, self.JtAng):
                for r in range(JP.shape[0]):
                    gl.glPushMatrix()
                    gl.glTranslate(JP[r, 0], JP[r, 1], JP[r, 2])
                    drawing.drawSolidSphere(radius=self.JtRadius, Color=[0, 0, 0, 0.6])
                    gl.glPopMatrix()
                    if self.showJointAng:
                        gl.glPushAttrib(gl.GL_LINE_BIT)
                        gl.glLineWidth(self.AxisWidth)
                        gl.glBegin(gl.GL_LINES)
                        Angle = np.linalg.norm(JA[r, :])
                        Axis = JA[r, :] / Angle
                        Start = JP[r, :] + self.AxisHalfLength*Axis
                        End = JP[r, :] - self.AxisHalfLength*Axis
                        gl.glVertex3f(Start[0], Start[1], Start[2])
                        gl.glVertex3f(End[0], End[1], End[2])
                        gl.glEnd()
                        gl.glPopAttrib()

        gl.glPopMatrix()

        if self.takeSS:
            x, y, width, height = gl.glGetIntegerv(gl.GL_VIEWPORT)
            # print("Screenshot viewport:", x, y, width, height)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)

            data = gl.glReadPixels(x, y, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            SS = np.frombuffer(data, dtype=np.uint8)
            SS = np.reshape(SS, (height, width, 4))
            SS = cv2.flip(SS, 0)
            SS = cv2.cvtColor(SS, cv2.COLOR_BGRA2RGBA)
            cv2.imwrite('screenshot_' + str(self.SSCtr).zfill(6) + '.png', SS)
            self.SSCtr = self.SSCtr + 1
            self.takeSS = False

            # Also serialize points
            for Idx, NOCS in enumerate(self.NOCS):
                if self.activeNMIdx != self.nNM:
                    if Idx != self.activeNMIdx:
                        continue
                NOCS.serialize('nocs_' + str(Idx).zfill(2) + '_' + str(self.SSCtr).zfill(6) + '.obj')
            print('[ INFO ]: Done saving.')
            sys.stdout.flush()

    def keyPressEvent(self, a0: QKeyEvent):
        if a0.key() == QtCore.Qt.Key_Plus:  # Increase or decrease point size
            if self.PointSize < 20:
                self.PointSize = self.PointSize + 1

        if a0.modifiers() != QtCore.Qt.NoModifier:
            return

        if a0.key() == QtCore.Qt.Key_Minus:  # Increase or decrease point size
            if self.PointSize > 1:
                self.PointSize = self.PointSize - 1

        if a0.key() == QtCore.Qt.Key_T:  # Toggle NOCS views
            if self.nNM > 0:
                self.activeNMIdx = (self.activeNMIdx + 1) % (self.nNM + 1)

        if a0.key() == QtCore.Qt.Key_N:
            self.showNOCS = not self.showNOCS
        if a0.key() == QtCore.Qt.Key_B:
            self.showBB = not self.showBB
        if a0.key() == QtCore.Qt.Key_M:
            self.showOBJModels = not self.showOBJModels
        if a0.key() == QtCore.Qt.Key_P:
            self.showPoints = not self.showPoints
        if a0.key() == QtCore.Qt.Key_W:
            self.showWireFrame = not self.showWireFrame
        if a0.key() == QtCore.Qt.Key_J:
            self.showJointPos = not self.showJointPos
            self.showJointAng = not self.showJointAng
        if a0.key() == QtCore.Qt.Key_A:
            self.showAnimation = not self.showAnimation
        if a0.key() == QtCore.Qt.Key_S:
            print('[ INFO ]: Taking snapshot and saving active NOCS maps as OBJ. This might take a while...')
            sys.stdout.flush()
            self.takeSS = True


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = Easel([StrobeNetModule()], sys.argv[1:])
    mainWindow.show()
    sys.exit(app.exec_())
