from utils import generate_gradcam_heatmaps, generate_bcos_explanations, torch_radon_transform

import gc
import time

import torch
import torch.optim as optim

class ModelTrainer:
    
    def __init__(self, model, model_name, is_model_bcos, is_model_teacher, dataset_name, train_loader, validation_loader, image_size, num_classes, idx_to_tag, batch_size, lr, epochs, device):
        self.device = device

        self.model = model
        self.model.to(self.device)
        
        self.model_name = model_name

        if is_model_bcos:
            self.model_prefix = "bcos_"
        else:
            self.model_prefix = ""

        if is_model_teacher:
            self.model_type = "teacher"
        else:
            self.model_type = "student"
        
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.image_size = image_size
        self.num_classes = num_classes
        self.idx_to_tag = idx_to_tag
        self.batch_size = batch_size

        self.lr = lr
        self.epochs = epochs
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.kl_divergence_loss = torch.nn.KLDivLoss(reduction='batchmean')

        self.alpha = 0.5
        self.epsilon=1e-10
        self.temperature_soft_labels = 10
    
    def _generate_soft_labels(self, logits, logits_from):
        if logits_from == 'teacher':
            return torch.nn.functional.softmax(logits / self.temperature_soft_labels, dim=1)
        elif logits_from == 'student':
            return torch.nn.functional.log_softmax(logits / self.temperature_soft_labels, dim=1)
        else:
            raise ValueError(f"please input correct source of logits. you entered {logits_from}, which is incorrect.")

    def _compute_distillation_loss(self, y_true, y_pred, y_soft_student, y_soft_teacher):
        loss_hard = self.criterion(y_pred, y_true)
        loss_soft = self.kl_divergence_loss(y_soft_student, y_soft_teacher) * (self.temperature_soft_labels ** 2)
        return (self.alpha * loss_hard) + ((1 - self.alpha) * loss_soft)
    
    def _compute_cosine_similarity_loss(self, teacher_heatmaps, student_heatmaps):        
        return (1 - self.alpha) * torch.mean(1 - torch.mean(torch.nn.functional.cosine_similarity(student_heatmaps, teacher_heatmaps), dim=1, keepdim=True))
    
    def _compute_wasserstein_loss(self, teacher_heatmaps, student_heatmaps):
        theta = torch.pi * torch.arange(180) / 180

        radon_teacher_heatmaps = torch_radon_transform(teacher_heatmaps, theta, self.device).cumsum(dim=1)
        radon_student_heatmaps = torch_radon_transform(student_heatmaps, theta, self.device).cumsum(dim=1)

        wasserstein_loss = torch.mean(1 - torch.mean(torch.nn.functional.cosine_similarity(radon_student_heatmaps, radon_teacher_heatmaps), dim=1, keepdim=True))

        del radon_teacher_heatmaps
        del radon_student_heatmaps
        gc.collect()

        return (1 - self.alpha) * wasserstein_loss
    
    # def _compute_wasserstein_loss(self, teacher_heatmaps, student_heatmaps):
    #     radon_cdt_teacher = self.radon_cdt.transform(teacher_heatmaps / (teacher_heatmaps.sum(dim=(1, 2), keepdim=True) + self.epsilon))
    #     radon_cdt_student = self.radon_cdt.transform(student_heatmaps / (student_heatmaps.sum(dim=(1, 2), keepdim=True) + self.epsilon))
    #     wasserstein_loss = torch.mean(torch.norm(radon_cdt_teacher - radon_cdt_student, dim=-1))
    #     return (1 - self.alpha) * wasserstein_loss

    def _validate(self):
        self.model.eval()

        validation_loss, validation_accuracy = 0.0, 0.0
        with torch.no_grad():
            for images, labels in self.validation_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
                validation_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                validation_accuracy += torch.sum(preds == labels.data).double()

        validation_loss /= len(self.validation_loader)
        validation_accuracy /= len(self.validation_loader.dataset)

        print(f"validation loss: {validation_loss:.2f} | validation accuracy: {validation_accuracy:.2f}", end=" | ")
        
        del validation_loss
        del validation_accuracy
        gc.collect()
        
        return None

    def train(self):
        for epoch in range(self.epochs):
            start_time = time.time()
            print(f"Epoch {epoch + 1} / {self.epochs}", end=" | ")

            running_loss, running_accuracy = 0.0, 0.0
            for images, labels in self.train_loader:
                torch.cuda.empty_cache()
                self.model.train()
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_accuracy += torch.sum(preds == labels.data).double()

                del outputs
                del preds
                gc.collect()

            print(f"training loss: {(running_loss / len(self.train_loader)):.2f}", end=" | ")
            self._validate()
            end_time = time.time()
            print(f"time elapsed: {(end_time - start_time) / 60:.2f} minutes")
                
            gc.collect()
        
        torch.save(self.model.state_dict(), f"./../models/{self.dataset_name}_{self.model_type}_{self.model_prefix+self.model_name}.pth")
        print(f"model training finshed. saving model to:  ./../models/{self.dataset_name}_{self.model_type}_{self.model_prefix+self.model_name}.pth")
        return None
        
    def train_simple_kd(self, teacher_model):
        teacher_model.eval()
        teacher_model = teacher_model.to(self.device)
        
        for epoch in range(self.epochs):
            start_time = time.time()
            print(f"Epoch {epoch + 1} / {self.epochs}", end=" | ")
            
            running_loss, running_accuracy = 0.0, 0.0
            for images, labels in self.train_loader:
                torch.cuda.empty_cache()
                self.model.train()
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                student_outputs = self.model(images)
                student_soft_labels = self._generate_soft_labels(student_outputs, 'student')

                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                teacher_soft_labels = self._generate_soft_labels(teacher_outputs, 'teacher')
                
                loss = self._compute_distillation_loss(labels, student_outputs, student_soft_labels, teacher_soft_labels)
                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()

                _, preds = torch.max(student_outputs, 1)
                running_accuracy += torch.sum(preds == labels.data).double()

                del student_outputs
                del student_soft_labels
                del teacher_outputs
                del teacher_soft_labels
                del preds
                gc.collect()

            print(f"training loss: {(running_loss / len(self.train_loader)):.2f}", end=" | ")
            self._validate()
            end_time = time.time()
            print(f"time elapsed: {(end_time - start_time) / 60:.2f} minutes")

            gc.collect()

        torch.save(self.model.state_dict(), f"./../models/{self.dataset_name}_{self.model_type}_{self.model_prefix+self.model_name}_simple_kd.pth")
        print(f"model training finshed. saving model to:  ./../models/{self.dataset_name}_{self.model_type}_{self.model_prefix+self.model_name}_simple_kd.pth")
        return None

    def train_explanable_kd(self, teacher_model_name, teacher_model, expkd_loss_type):
        # self.radon_cdt = RadonCDT(image_shape=self.image_size[0], epsilon=self.epsilon)
        teacher_model = teacher_model.to(self.device)

        for epoch in range(self.epochs):
            start_time = time.time()
            print(f"Epoch {epoch + 1} / {self.epochs}", end=" | ")
            
            running_loss, running_accuracy = 0.0, 0.0
            for images, labels in self.train_loader:
                torch.cuda.empty_cache()
                self.model.train()
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                student_outputs = self.model(images)
                student_soft_labels = self._generate_soft_labels(student_outputs, 'student')
                
                if self.model_prefix == 'bcos_':
                    student_heatmaps = generate_bcos_explanations(self.model, images, labels)
                else:
                    student_heatmaps = generate_gradcam_heatmaps(self.model, self.model_name, images, labels, self.image_size)
                
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                teacher_soft_labels = self._generate_soft_labels(teacher_outputs, 'teacher')
                
                if self.model_prefix == 'bcos_':
                    teacher_heatmaps = generate_bcos_explanations(teacher_model, images, labels)
                else:
                    teacher_heatmaps = generate_gradcam_heatmaps(teacher_model, teacher_model_name, images, labels, self.image_size)
                
                distillation_loss = self._compute_distillation_loss(labels, student_outputs, student_soft_labels, teacher_soft_labels)
                
                if expkd_loss_type == 'csm':
                    heatmap_loss = self._compute_cosine_similarity_loss(teacher_heatmaps, student_heatmaps)
                elif expkd_loss_type == 'swd':
                    heatmap_loss = self._compute_wasserstein_loss(teacher_heatmaps, student_heatmaps)

                loss = distillation_loss + heatmap_loss

                del student_soft_labels
                del student_heatmaps
                del teacher_outputs
                del teacher_soft_labels
                del teacher_heatmaps
                gc.collect()
                
                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()

                _, preds = torch.max(student_outputs, 1)
                running_accuracy += torch.sum(preds == labels.data).double()
                
                del student_outputs
                del loss
                del preds
                gc.collect()
                            
            print(f"training loss: {(running_loss / len(self.train_loader)):.2f}", end=" | ")
            self._validate()
            end_time = time.time()
            print(f"time elapsed: {(end_time - start_time) / 60:.2f} minutes")

            gc.collect()

        torch.save(self.model.state_dict(), f"./../models/{self.dataset_name}_{self.model_type}_{self.model_prefix+self.model_name}_explainable_kd_{expkd_loss_type}.pth")
        print(f"model training finshed. saving model to:  ./../models/{self.dataset_name}_{self.model_type}_{self.model_prefix+self.model_name}_explainable_kd_{expkd_loss_type}.pth")
        return None